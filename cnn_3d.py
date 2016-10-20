import tensorflow as tf
import numpy as np
import h5py

DATA_DIR = 'data'

class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 100

def _load_from_h5(filename):
    f = h5py.File(filename, 'r')
    data = f['data'][:]
    f.close()
    return data

class CNN_3D(object):
    
    def load_data(self):
        self.train_images = _load_from_h5(DATA_DIR + '/train_images.h5')
        self.val_images = _load_from_h5(DATA_DIR + '/val_images.h5')

        self.train_labels = _load_from_h5(DATA_DIR + '/train_labels.h5')
        self.val_labels = _load_from_h5(DATA_DIR + '/val_labels.h5')

    def inference():
        pass

    def run_epoch():


    def __init__(self, config):
        self.config = config




