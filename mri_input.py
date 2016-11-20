import tensorflow as tf
import numpy as np

IMAGE_DIMS = [96, 112, 96]

def distort_image(image):

    distorted_image = tf.image.random_flip_left_right(image)

    distorted_image = tf.image.random_contrast(distorted_image)


    return image


# Adapted from https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
def read_and_decode_single_example(filename, train=True):
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
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

    # distort image
    if train:
        image = distort_image(image)

    return image, label

if __name__ == '__main__':
    read_and_decode_single_example('data/mri.tfrecords')
