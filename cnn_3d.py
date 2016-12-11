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

class Config():

    lr = 0.001
    l2 = 0.01
    dropout = 0.8

    kernel_size = 3 
    num_filters = 8
    downsample_every = 2

    num_layers = 4
    num_layers_to_train = num_layers
    num_layers_to_restore = 0
    downsample_every = 2
    downsample_factor = 2
    use_sex_labels = False

    mode = 'pretrain'


class CNN_3D(object):

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

    def calc_loss(self, outputs, labels):

        # in pretraining outputs and labels are the full 3d MRIs
        if self.config.mode == 'pretrain':
            loss = tf.reduce_sum(tf.square(outputs - labels))
        else:
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, labels)) 
        loss += tf.reduce_sum(tf.pack(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))


        return loss

    def add_train_op(self, loss):
        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        return train_op

    def correlation_inference(self, correlation):
        flattened = slim.flatten(correlation)
        #summed = tf.reduce_sum(flattened)
        #flattened = tf.Print(flattened, [summed])
        output = slim.fully_connected(flattened, 2, weights_regularizer=slim.l2_regularizer(self.config.l2))
        return output

    def image_inference(self, images, train=True):

        images = tf.expand_dims(images, -1)

        # for convenience
        num_filters = self.config.num_filters
        kernel_size = self.config.kernel_size
        downsample_every = self.config.downsample_every
        num_layers = self.config.num_layers
        num_layers_to_train = self.config.num_layers_to_train

        # conv
        if num_layers > 0:
            trainable = True if num_layers_to_train >= num_layers else False
            forward = conv3d(images, kernel_size, 1, num_filters,
                    scope='conv_1', trainable=trainable)
        else:
            forward = images

        for i in range(num_layers - 1):
            stride = 2 if i % downsample_every == 0 else 1
            trainable = True if i+2 > num_layers - num_layers_to_train else False
            forward = conv3d(forward, kernel_size, num_filters, num_filters,
                    scope='conv_' + str(i+2), stride=stride, trainable=trainable)
            if stride == 2:
                forward = slim.dropout(forward, keep_prob=self.config.dropout, is_training=train)

        if self.config.mode == 'pretrain':
            backward = forward
            # deconv
            for i in range(num_layers - 1):
                stride = 1 if i % downsample_every == 0 else 2
                backward = conv3d_transpose(backward, kernel_size, num_filters, num_filters,
                        scope='conv_' + str(num_layers - i), stride=stride)
            backward = conv3d_transpose(backward, kernel_size, num_filters, 1, scope='conv_1', stride=2)
            output = tf.squeeze(backward)
        else:

            flattened = slim.flatten(forward)
        
            with tf.variable_scope('fully_connected'):
                # fully connected layer
                output = slim.fully_connected(flattened, 2000, weights_regularizer=slim.l2_regularizer(self.config.l2))
                output = slim.fully_connected(output, 500, weights_regularizer=slim.l2_regularizer(self.config.l2))
                output = slim.fully_connected(output, 2, activation_fn=None, weights_regularizer=slim.l2_regularizer(self.config.l2))

        return output, forward

    def inference(self, images, correlation, train):
        if self.config.use_correlation == 2: 
            corr_outputs = self.correlation_inference(correlation)
            return corr_outputs
        image_outputs, self.filt = self.image_inference(images, train)
        return image_outputs

    def add_summaries(self, images, train):
        dataset = 'train' if train else 'validation'
        tf.scalar_summary('loss', self.loss)
        tf.scalar_summary('accuracy', self.accuracy)
        if self.config.use_correlation != 2:
            filt_depth = int(self.filt.get_shape()[1])/2
            im_depth = int(images.get_shape()[1])/2
            filt = tf.squeeze(
                    tf.slice(self.filt, [0, filt_depth, 0, 0, 0], [-1, 1, -1, -1, 1])
                    , [1])
            image = tf.squeeze(
                    tf.slice(images, [0, im_depth, 0, 0], [-1, 1, -1, -1])
                    )
            tf.image_summary('filter', filt)
            tf.image_summary('image', tf.expand_dims(image, -1))

    def __init__(self, config, image_batch, label_batch, corr_batch, train=True):
        self.config = config
        label_batch = image_batch if config.mode == 'pretrain' else label_batch
        self.outputs = self.inference(image_batch, corr_batch, train)
        self.loss = self.calc_loss(self.outputs, label_batch)
        self.predictions = self.make_predictions(self.outputs)
        self.accuracy = self.calc_accuracy(self.predictions, label_batch)
        self.train_op = self.add_train_op(self.loss)
        self.add_summaries(image_batch, train)
        self.merged = tf.merge_all_summaries()
        

