import tensorflow as tf
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from nn_utils import conv3d, conv3d_transpose

import sys
import time

slim = tf.contrib.slim

DATA_DIR = 'data'

LR = 0.0001
L2 = 0.001
MAX_EPOCHS = 100
EARLY_STOPPING = 20
DROPOUT = 0.9

KERNEL_SIZE = 3 
NUM_FILTERS = 8
DOWNSAMPLE_EVERY = 4

MODE = 'pretrain'

class 3D_ResNet(3D_CNN):

    def inference(images, num_layers, num_layers_to_train, mode='pretrain', train=True):

        images = tf.expand_dims(images, -1)

        # conv
        trainable = True if num_layers_to_train >= num_layers else False
        forward = conv3d(images, KERNEL_SIZE, 1, NUM_FILTERS,
                scope='conv_1', trainable=trainable)

        for i in range(num_layers - 1):
            stride = 2 if i % DOWNSAMPLE_EVERY == 0 else 1
            trainable = True if i+2 > num_layers - num_layers_to_train else False
            forward = conv3d(forward, KERNEL_SIZE, NUM_FILTERS, NUM_FILTERS,
                    scope='conv_' + str(i+2), stride=stride, trainable=trainable)
            if stride == 2:
                forward = slim.dropout(forward, keep_prob=DROPOUT, is_training=train)

        if mode == 'pretrain':
            backward = forward
            # deconv
            for i in range(num_layers - 1):
                stride = 1 if i % DOWNSAMPLE_EVERY == 0 else 2
                backward = conv3d_transpose(backward, KERNEL_SIZE, NUM_FILTERS, NUM_FILTERS,
                        scope='conv_' + str(num_layers - i), stride=stride)
            backward = conv3d_transpose(backward, KERNEL_SIZE, NUM_FILTERS, 1, scope='conv_1', stride=2)
            output = tf.squeeze(backward)
        else:

            flattened = slim.flatten(forward)
        
            with tf.variable_scope('fully_connected'):
                # fully connected layer
                output = slim.fully_connected(flattened, 2000, weights_regularizer=slim.l2_regularizer(L2))
                output = slim.fully_connected(output, 500, weights_regularizer=slim.l2_regularizer(L2))
                output = slim.fully_connected(output, 2, activation_fn=None, weights_regularizer=slim.l2_regularizer(L2))

        return output, forward

