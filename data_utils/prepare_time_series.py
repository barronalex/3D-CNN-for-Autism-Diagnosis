import pandas as pd
import h5py
import numpy as np
import os
from tqdm import tqdm

DATA_DIR = 'data/time_series/'
NUM_BRAIN_REGIONS = 116

filenames = os.listdir(DATA_DIR)
corr_file = h5py.File('data/correlation.h5', 'w')

lengths = {}
eeg = {}
max_length = 0

for i, f in enumerate(tqdm(filenames)):
    id = f[:5]
    data = pd.read_csv(DATA_DIR + f, header=None).as_matrix()
    data = np.reshape(data, (NUM_BRAIN_REGIONS, len(data)/NUM_BRAIN_REGIONS))
    # find pearson correlation
    correlation = np.corrcoef(data)
    correlation = np.nan_to_num(correlation)
    corr_file.create_dataset(id, data = correlation)
    eeg[id] = data
    
    if data.shape[1] > max_length:
        max_length = data.shape[1]

    
    
print max_length
eeg_out = []
lengths_out = []
labels_out = []

phenotype_data = pd.read_csv('data/phenotype_data.csv', header=None)
phenotype_data = phenotype_data.as_matrix()
labels = np.array(phenotype_data[:,2] - 1, dtype=int)
ids = np.array(phenotype_data[:,1], dtype=str)

# pad the eeg_files
for i, id in enumerate(ids):
    if id not in eeg: continue
    eg = eeg[id]
    length = eg.shape[1]
    eg = np.pad(eg, ((0, 0), (0, max_length - length)), 'constant', constant_values=(0))

    print eg.shape

    label = labels[i]

    eeg_out.append(eg)
    labels_out.append(label)
    lengths_out.append(length)

# must be multiples of 10
train_split = 0.7
val_split = 0.1
test_split = 0.2

round_num_examples = len(eeg_out) - len(eeg_out) % 10
train_num = int(round_num_examples * train_split)
val_num = int(round_num_examples * val_split)
test_num = int(round_num_examples * test_split + len(eeg_out) % 10)
print train_num, val_num, test_num, len(eeg_out)

def save_to_h5(files, name, data):
    train = data[:train_num]
    val = data[train_num:train_num+val_num]
    test = data[:test_num]
    files[0].create_dataset(name, data = train)
    files[1].create_dataset(name, data = val)
    files[2].create_dataset(name, data = test)
    

train_file = h5py.File('data/eeg/train.h5', 'w')
val_file = h5py.File('data/eeg/val.h5', 'w')
test_file = h5py.File('data/eeg/test.h5', 'w')

save_to_h5([train_file, val_file, test_file], 'eeg', np.stack(eeg_out))
save_to_h5([train_file, val_file, test_file], 'labels', np.stack(labels_out))
save_to_h5([train_file, val_file, test_file], 'lengths', np.stack(lengths_out))

train_file.close()
val_file.close()
test_file.close()
corr_file.close()
