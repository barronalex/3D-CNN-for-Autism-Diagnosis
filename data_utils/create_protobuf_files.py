import tensorflow as tf
import numpy as np
import h5py
from tqdm import tqdm

DATA_DIR = '../data'

# structure from https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

def load_from_h5(filename):
    f = h5py.File(filename, 'r')
    data = f['data'][:]
    f.close()
    return data

# load files from h5py
train_images = load_from_h5(DATA_DIR + '/train_images.h5')
train_labels = load_from_h5(DATA_DIR + '/train_labels.h5')

def save_to_record(images, labels, split='train'):

    writer = tf.python_io.TFRecordWriter(DATA_DIR + '/mri' + '_' + split + '.tfrecords')

    for i in tqdm(range(len(train_images))):
        image = train_images[i]
        label = train_labels[i]

        example = tf.train.Example(
                features = tf.train.Features(
                    feature ={
                        'label': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[label])),
                        'image': tf.train.Feature(
                            float_list=tf.train.FloatList(value=image.flatten())
                            )
                    }
                )
            )
        serialized = example.SerializeToString()
        writer.write(serialized)



