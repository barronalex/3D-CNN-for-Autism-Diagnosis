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
DROPOUT = 0.8

KERNEL_SIZE = 3 
NUM_FILTERS = 8
DOWNSAMPLE_EVERY = 2

MODE = 'pretrain'

class CNN_3D():

    def _save_images(self, images, outputs, name):
        if len(outputs.shape) == 4:
            outputs = np.expand_dims(outputs, 4)
        im_depth = images.shape[1]/2
        out_depth = outputs.shape[1]/2
        fig = plt.figure()
        a = fig.add_subplot(1,2,1)
        imgplot = plt.imshow(images[0,im_depth,:,:])
        a.set_title('Original')
        a = fig.add_subplot(1,2,2)
        imgplot = plt.imshow(outputs[0,out_depth,:,:,0])
        a.set_title('Reconstructed')
        fig.savefig('figures/' + name + ".png")
        plt.close()

    def make_predictions(self, output):
        """Get answer predictions from output"""
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred

    def calc_accuracy(self, predictions, labels):
        correct_prediction = tf.equal(predictions, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def calc_loss(self, outputs, labels, mode='pretrain'):

        # in pretraining outputs and labels are the full 3d MRIs
        if mode == 'pretrain':
            loss = tf.reduce_sum(tf.square(outputs - labels))
        else:
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, labels)) 
        loss += tf.reduce_sum(tf.pack(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))


        return loss

    def add_train_op(self, loss):
        train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)
        return train_op

    def inference(self, images, num_layers, num_layers_to_train, mode='pretrain', train=True):

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

    def add_summaries(self, images, train):
        dataset = 'train' if train else 'validation'
        tf.scalar_summary('loss', self.loss)
        tf.scalar_summary('accuracy', self.accuracy)
        filt_depth = int(self.filt.get_shape()[1])/2
        im_depth = int(images.get_shape()[1])/2
        filt = tf.squeeze(
                tf.slice(self.filt, [0, filt_depth, 0, 0, 0], [-1, 1, -1, -1, 1])
                , [3])
        image = tf.squeeze(
                tf.slice(images, [0, im_depth, 0, 0], [-1, 1, -1, -1])
                )
        tf.image_summary('filter', filt)
        tf.image_summary('image', tf.expand_dims(image, -1))

    def __init__(self, image_batch, label_batch, num_layers, mode, num_layers_to_train=0, train=True):
        label_batch = image_batch if mode == 'pretrain' else label_batch
        self.outputs, self.filt = self.inference(image_batch, num_layers, num_layers_to_train, mode, train)
        self.loss = self.calc_loss(self.outputs, label_batch, mode)
        self.predictions = self.make_predictions(self.outputs)
        self.accuracy = self.calc_accuracy(self.predictions, label_batch)
        self.train_op = self.add_train_op(self.loss)
        self.add_summaries(image_batch, train)
        self.merged = tf.merge_all_summaries()
        

