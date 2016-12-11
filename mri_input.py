import tensorflow as tf
import numpy as np

from nn_utils import rotate_image_tensor

IMAGE_DIMS = [96, 112, 96]
NOISE_STD = 0.1


def distort_image(image, rotate, noise):

    # randomly choose an axis of rotation
    axis = 0

    if rotate:
        image = tf.expand_dims(image, -1)

        image = tf.split(axis, image.get_shape()[axis].value, image)
        image = [tf.squeeze(im, squeeze_dims=[axis]) for im in image]

        # rotate image by a random angle between 0 and pi
        angle = tf.random_uniform((), 0, 2*3.141)
        image = [rotate_image_tensor(im, angle) for im in image]

        image = tf.pack(image)
        image = tf.squeeze(image)

    noise = tf.random_normal(image.get_shape(), stddev=noise)
    image += noise
    

    return image

def process_image(image, downsample_factor, train, rotate, noise):
    image = tf.reshape(image, IMAGE_DIMS)

    if downsample_factor > 1:
        image = tf.expand_dims(image, -1)
        image = tf.expand_dims(image, 0)
        image = tf.nn.avg_pool3d(image, 
                [1,downsample_factor,downsample_factor,downsample_factor,1],
                [1,downsample_factor,downsample_factor,downsample_factor,1],
                'SAME')
        image = tf.squeeze(image)

    if train:
        image = distort_image(image, rotate, noise)
    return image


# Adapted from https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
def read_and_decode_single_example(filename_queue, train=True,
        downsample_factor=1, corr=0, rotate=True, noise=0):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'sex': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature(np.prod(IMAGE_DIMS), tf.float32),
            'corr': tf.FixedLenFeature(116**2, tf.float32)
        })
    # now return the converted data
    label = features['label']
    sex = features['sex']
    corr = features['corr']
    corr = tf.reshape(corr, [116, 116])
    image = features['image']

    if corr != 2:
        image = process_image(image, downsample_factor, train, rotate, noise)

    return image, label, sex, corr
