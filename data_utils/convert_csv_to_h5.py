import pandas as pd
import h5py

DATA_DIR = 'data/'

def convert_csv_to_h5(filename):
    print 'converting', filename
    data_frame = pd.read_csv(DATA_DIR + filename, header=None)
    h5_file = h5py.File(DATA_DIR + filename[:-3] + 'h5', 'w')
    h5_file.create_dataset('data', data = data_frame.as_matrix())
    h5_file.close()

for fn in ['fALFF.csv', 'coord.csv', 'region_code.csv']:
    convert_csv_to_h5(DATA_DIR + fn)
