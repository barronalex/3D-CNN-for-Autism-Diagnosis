import h5py
import numpy as np
import pandas as pd

DATA_DIR = 'data/'

image_dimensions = [91, 109, 91]

# must be multiples of 10
train_split = 0.7
val_split = 0.1
test_split = 0.2

assert train_split + val_split + test_split == 1

def save_to_h5(filename, d):
    o = h5py.File(DATA_DIR + filename, 'w')
    o.create_dataset('data', data = d)
    o.close()

# load voxel value and coordinate files
f = h5py.File(DATA_DIR + 'fALFF.h5', 'r')
c = h5py.File(DATA_DIR + 'coord.h5', 'r')

fALFF = f['data'][:]
coord = c['data'][:]

# normalize to unit mean and variance
fALFF = ((fALFF.T - np.mean(fALFF, axis=1)) / np.std(fALFF, axis=1)).T

f.close()
c.close()

round_num_examples = fALFF.shape[0] - fALFF.shape[0] % 10
train_num = int(round_num_examples * train_split)
val_num = int(round_num_examples * val_split)
test_num = int(round_num_examples * test_split + fALFF.shape[0] % 10)

#create empty images
images = np.zeros([fALFF.shape[0]] + image_dimensions)

for i in range(fALFF.shape[0]):
    if i % 100 == 0:
        print str(i) + '/' + str(fALFF.shape[0])
    for j in range(fALFF.shape[1]):
        images[tuple([i] + coord[j].tolist())] = fALFF[i, j]


images = np.pad(images, ((0,0),(5,0),(3,0),(5,0)), 'constant', constant_values=0)

train_images = images[:train_num]
val_images = images[train_num:train_num+val_num]
test_images = images[-test_num:]

phenotype_data = pd.read_csv(DATA_DIR + 'phenotype_data.csv', header=None)
phenotype_data = phenotype_data.as_matrix()

labels = np.array(phenotype_data[:,2] - 1, dtype=int)

train_labels = labels[:train_num]
val_labels = labels[train_num:train_num+val_num]
test_labels = labels[-test_num:]

save_to_h5('train_images.h5', train_images)
save_to_h5('val_images.h5', val_images)
save_to_h5('test_images.h5', test_images)

save_to_h5('train_labels.h5', train_labels)
save_to_h5('val_labels.h5', val_labels)
save_to_h5('test_labels.h5', test_labels)
