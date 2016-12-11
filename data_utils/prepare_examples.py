import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import argparse

import sys

DATA_DIR = 'data/'

image_dimensions = [91, 109, 91]

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gate', default='')
args = parser.parse_args()

# must be multiples of 10
train_split = 0.7
val_split = 0.1
test_split = 0.2

assert train_split + val_split + test_split == 1

def save_to_record(images, labels, ages, sexes, ids, split='train'):
    print ''
    print '==> saving ' + split + ' data into tf records file'
    writer = tf.python_io.TFRecordWriter(DATA_DIR + '/mri' + '_' + split + '.tfrecords')

    corr_file = h5py.File('data/correlation.h5', 'r')
    eeg_file = h5py.File('data/eeg.h5', 'r')

    for i in tqdm(range(len(images))):
        image = images[i]
        label = labels[i]
        age = ages[i]
        sex = sexes[i]

        if ids[i] not in corr_file: continue
        corr = corr_file[ids[i]][:]
        # pad to max_length
        

        example = tf.train.Example(
                features = tf.train.Features(
                    feature ={
                        'label': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[label])
                            ),
                        'image': tf.train.Feature(
                            float_list=tf.train.FloatList(value=image.flatten())
                            ),
                        'age': tf.train.Feature(
                            float_list=tf.train.FloatList(value=[age])
                            ),
                        'sex': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[sex])
                            ),
                        'corr': tf.train.Feature(
                            float_list=tf.train.FloatList(value=corr.flatten())
                            ),
                        'eeg': tf.train.Feature(
                            float_list=tf.train.FloatList(value=eeg.flatten())
                            )
                    }
                )
            )
        serialized = example.SerializeToString()
        writer.write(serialized)
    corr_file.close()

# load voxel value and coordinate files
f = h5py.File(DATA_DIR + 'fALFF.h5', 'r')
c = h5py.File(DATA_DIR + 'coord.h5', 'r')

fALFF = f['data'][:]
coord = c['data'][:]

# normalize to unit mean and variance
fALFF = ((fALFF.T - np.mean(fALFF, axis=1)) / np.std(fALFF, axis=1)).T

f.close()
c.close()


phenotype_data = pd.read_csv(DATA_DIR + 'phenotype_data.csv', header=None)
phenotype_data = phenotype_data.as_matrix()

# 0 is autistic, 1 is control
labels = np.array(phenotype_data[:,2] - 1, dtype=int)
ages = np.array(phenotype_data[:,4], dtype=float)
# 0 is male, 1 is female
sexes = np.array(phenotype_data[:,5] - 1, dtype=int)
ids = np.array(phenotype_data[:,1], dtype=str)

# create dataset with equal proportions of men and women
if args.gate == 'equal_gender':
    male = list(np.where(sexes == 0)[0])
    female = list(np.where(sexes == 1)[0])
    gated_indices = sorted(female + male[:len(female)])
elif args.gate == 'male':
    gated_indices = list(np.where(sexes == 0)[0])
else:
    gated_indices = range(fALFF.shape[0])

# for debugging
if args.gate == 'shuffle':
    p = np.random.permutation(fALFF.shape[0])
    labels = labels[p]
    sexes = sexes[p]
    gated_indices = list(np.where(sexes == 0)[0])


# create empty images
images = np.zeros([fALFF.shape[0]] + image_dimensions)

print ''
print '==> constructing full 3D images from coordinate values'
for i in tqdm(range(fALFF.shape[0])):
    for j in range(fALFF.shape[1]):
        images[tuple([i] + coord[j].tolist())] = fALFF[i, j]


images = np.pad(images, ((0,0),(5,0),(3,0),(5,0)), 'constant', constant_values=0)

round_num_examples = len(gated_indices) - len(gated_indices) % 10
train_num = int(round_num_examples * train_split)
val_num = int(round_num_examples * val_split)
test_num = int(round_num_examples * test_split + fALFF.shape[0] % 10)
print train_num, val_num, test_num, len(gated_indices)

np.random.seed(23) #for reproducibility
p = np.random.permutation(train_num + val_num)

def split_data(data):
    # randomly permute train and val data
    data = data[gated_indices]
    data[:len(p)] = data[p]
    return data[:train_num], data[train_num:train_num+val_num], data[train_num+val_num:train_num+val_num+test_num]

train_labels, val_labels, test_labels = split_data(labels)
train_ages, val_ages, test_ages = split_data(ages)
train_sexes, val_sexes, test_sexes = split_data(sexes)
train_ids, val_ids, test_ids = split_data(ids)
train_images, val_images, test_images = split_data(images)


save_to_record(train_images, train_labels, train_ages, train_sexes, train_ids, split=args.gate + 'train')
save_to_record(val_images, val_labels, val_ages, val_sexes, val_ids, split=args.gate + 'val')
save_to_record(test_images, test_labels, test_ages, test_sexes, test_ids, split=args.gate + 'test')
