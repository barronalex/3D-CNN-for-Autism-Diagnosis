import tensorflow as tf
import numpy as np

from nn_utils import rotate_image_tensor

IMAGE_DIMS = [96, 112, 96]


def distort_image(image):

    #distorted_image = tf.image.random_flip_left_right(image)
    #

    #distorted_image = tf.image.random_contrast(distorted_image)

    # let's start with random rotations and reflections
    # and scaling the saturation perhaps


    image = tf.expand_dims(image, -1)

    image = tf.split(0, int(image.get_shape()[0]), image)
    image = [tf.squeeze(im, squeeze_dims=[0]) for im in image]

    # rotate image by a random angle between 0 and pi
    angle = tf.random_uniform((), 0, 3.141)
    image = [rotate_image_tensor(im, angle) for im in image]

    image = tf.pack(image)
    image = tf.squeeze(image)

    return image


# Adapted from https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
def read_and_decode_single_example(filename_queue, train=True, downsample_factor=1):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature(np.prod(IMAGE_DIMS), tf.float32)
        })
    # now return the converted data
    label = features['label']
    image = features['image']
    image = tf.reshape(image, IMAGE_DIMS)

    if downsample_factor > 1:
        image = tf.nn.avg_pool3d(image, 
                [1,downsample_factor,downsample_factor,downsample_factor,1],
                [1,downsample_factor,downsample_factor,downsample_factor,1],
                'SAME')

    if train:
        image = distort_image(image)

    return image, label

if __name__ == '__main__':
    read_and_decode_single_example('data/mri.tfrecords')
