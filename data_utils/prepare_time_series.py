import pandas as pd
import h5py
import numpy as np
import os
from tqdm import tqdm

DATA_DIR = 'data/time_series/'
NUM_BRAIN_REGIONS = 116

filenames = os.listdir(DATA_DIR)
output_file = h5py.File('data/time_series.h5', 'w')

for f in tqdm(filenames):
    id = f[:5]
    data = pd.read_csv(DATA_DIR + f, header=None).as_matrix()
    data = np.reshape(data, (NUM_BRAIN_REGIONS, len(data)/NUM_BRAIN_REGIONS))
    # find pearson correlation
    correlation = np.corrcoef(data)
    output_file.create_dataset(id, data = correlation)

output_file.close()
