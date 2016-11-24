import tensorflow as tf
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from conv_utils import conv3d, conv3d_transpose

import sys
import time

slim = tf.contrib.slim

DATA_DIR = 'data'

LR = 0.001
L2 = 0.0001
HIDDEN_SIZE = 100
MAX_EPOCHS = 100
EARLY_STOPPING = 20
DROPOUT = 0.8

KERNEL_SIZE = 3 
NUM_FILTERS = 8

MODE = 'pretrain'

def _save_images(images, outputs, name, depth=50):
    for i in range(2):
        fig = plt.figure()
        a = fig.add_subplot(1,2,1)
        imgplot = plt.imshow(images[i,depth,:,:])
        a.set_title('Original')
        a = fig.add_subplot(1,2,2)
        imgplot = plt.imshow(outputs[i,depth,:,:])
        a.set_title('Reconstructed')
        fig.savefig('figures/' + name + str(i) + ".png")
        plt.close()

def predictions(output):
    """Get answer predictions from output"""
    preds = tf.nn.softmax(output)
    pred = tf.argmax(preds, 1)
    return pred

def loss(outputs, labels, mode='pretrain'):

    # in pretraining outputs and labels are the full 3d MRIs
    if mode == 'pretrain':
        loss = tf.reduce_sum(tf.square(outputs - labels))
    else:
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, labels)) 
    loss += tf.reduce_sum(tf.pack(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

    return loss

def train_op(loss):
    train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)
    return train_op

def inference(images, mode='pretrain'):

    if mode == 'pretrain':
        trainable = True
    else:
        trainable = False

    #for i in range(self.config.num_layers):
        #stride = 1 if i % 2 == 0 else 2
        #forward = conv3d(forward, 3, 15, 15, scope='conv_2', stride=stride, trainable=trainable)

    images = tf.expand_dims(images, -1)

    forward1 = conv3d(images, KERNEL_SIZE, 1, NUM_FILTERS, scope='conv_1', trainable=trainable)
    forward2 = conv3d(forward1, KERNEL_SIZE, NUM_FILTERS, NUM_FILTERS, scope='conv_2', stride=2, trainable=trainable)

    forward2 = slim.dropout(forward2)

    forward3 = conv3d(forward2, KERNEL_SIZE, NUM_FILTERS, NUM_FILTERS, scope='conv_3', trainable=trainable)
    forward4 = conv3d(forward3, KERNEL_SIZE, NUM_FILTERS, NUM_FILTERS, scope='conv_4', stride=2, trainable=trainable)

    forward4 = slim.dropout(forward4)

    print forward4.get_shape()

    #forward5 = conv3d(forward4, 3, NUM_FILTERS, NUM_FILTERS, scope='conv_5', trainable=trainable)
    #forward6 = conv3d(forward5, 3, NUM_FILTERS, NUM_FILTERS, scope='conv_6', stride=2, trainable=trainable)

    #forward6 = slim.dropout(forward6)

    if mode == 'pretrain':
        #backward1 = conv3d_transpose(forward6, 3, NUM_FILTERS, NUM_FILTERS, scope='conv_6')
        #backward2 = conv3d_transpose(backward1, 3, NUM_FILTERS, NUM_FILTERS, scope='conv_5', stride=2)

        backward3 = conv3d_transpose(forward4, KERNEL_SIZE, NUM_FILTERS, NUM_FILTERS, scope='conv_4')
        backward4 = conv3d_transpose(backward3, KERNEL_SIZE, NUM_FILTERS, NUM_FILTERS, scope='conv_3', stride=2)

        backward5 = conv3d_transpose(backward4, KERNEL_SIZE, NUM_FILTERS, NUM_FILTERS, scope='conv_2')
        backward6 = conv3d_transpose(backward5, KERNEL_SIZE, NUM_FILTERS, 1, scope='conv_1', stride=2)

        output = tf.squeeze(backward6)

    else:
        flattened = slim.flatten(forward4)
    
        print flattened.get_shape()

        with tf.variable_scope('fully_connected'):
            # fully connected layer
            output = slim.fully_connected(flattened, 500, weights_regularizer=slim.l2_regularizer(L2))
            output = slim.fully_connected(output, 500, weights_regularizer=slim.l2_regularizer(L2))
            output = slim.fully_connected(output, 2, activation_fn=None, weights_regularizer=slim.l2_regularizer(L2))

    return output

