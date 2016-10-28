import tensorflow as tf
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from conv_utils import conv3d, conv3d_transpose

import sys
import time

slim = tf.contrib.slim

image_dimensions = [96, 112, 96]
DATA_DIR = 'data'


class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 10
    lr = 0.001
    l2 = 0.0001
    hidden_size = 100
    max_epochs = 100
    early_stopping = 20
    dropout = 0.8

    mode = 'pretrain'

def _load_from_h5(filename):
    f = h5py.File(filename, 'r')
    data = f['data'][:]
    f.close()
    return data


def _save_images(images, outputs, name, depth=5):
    for i in range(images.shape[0]):
        fig = plt.figure()
        a = fig.add_subplot(1,2,1)
        imgplot = plt.imshow(images[i,depth,:,:,0])
        a.set_title('Original')
        plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation ='horizontal')
        a = fig.add_subplot(1,2,2)
        imgplot = plt.imshow(outputs[i,depth,:,:,0])
        imgplot.set_clim(0.0,0.7)
        a.set_title('Reconstructed')
        plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation ='horizontal')
        fig.savefig('figures/' + name + str(i) + ".png")

class CNN_3D(object):
    
    def load_data(self):
        print '<== loading train/val data'
        train_images = _load_from_h5(DATA_DIR + '/train_images.h5')
        val_images = _load_from_h5(DATA_DIR + '/val_images.h5')

        train_images = np.expand_dims(train_images, axis=-1)
        val_images = np.expand_dims(val_images, axis=-1)

        train_labels = _load_from_h5(DATA_DIR + '/train_labels.h5')
        val_labels = _load_from_h5(DATA_DIR + '/val_labels.h5')

        self.train = train_images, train_labels
        self.val = val_images, val_labels

    def add_placeholders(self):
        self.f_images_placeholder = tf.placeholder(tf.float32, shape=[self.config.batch_size] + image_dimensions + [1])
        self.autism_labels_placeholder = tf.placeholder(tf.int64, shape=(self.config.batch_size,))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def get_predictions(self, output):
        """Get answer predictions from output"""
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred

    def add_loss_op(self, output):

        if self.config.mode == 'pretrain':
            loss = tf.reduce_sum(tf.square(output - self.f_images_placeholder))
        else:
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output, self.autism_labels_placeholder)) 
        loss += tf.reduce_sum(tf.pack(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

        print len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        return loss

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)

        return train_op

    def inference(self):

        if self.config.mode == 'pretrain':
            trainable = True
        else:
            trainable = True

        images = slim.dropout(self.f_images_placeholder, keep_prob=self.dropout_placeholder)

        forward1 = conv3d(images, 3, 1, 15, scope='conv_1', trainable=trainable)
        forward2 = conv3d(forward1, 3, 15, 15, scope='conv_2', stride=2, trainable=trainable)

        forward2 = slim.dropout(forward2)

        forward3 = conv3d(forward2, 3, 15, 15, scope='conv_3', trainable=trainable)
        forward4 = conv3d(forward3, 3, 15, 15, scope='conv_4', stride=2, trainable=trainable)

        forward4 = slim.dropout(forward4)

        forward5 = conv3d(forward4, 3, 15, 15, scope='conv_5', trainable=trainable)
        forward6 = conv3d(forward5, 3, 15, 15, scope='conv_6', stride=2, trainable=trainable)

        forward6 = slim.dropout(forward6)

        self.forward = forward6

        if self.config.mode == 'pretrain':
            backward1 = conv3d_transpose(forward6, 3, 15, 15, scope='conv_6')
            backward2 = conv3d_transpose(backward1, 3, 15, 15, scope='conv_5', stride=2)

            backward3 = conv3d_transpose(forward4, 3, 15, 15, scope='conv_4')
            backward4 = conv3d_transpose(backward3, 3, 15, 15, scope='conv_3', stride=2)

            backward5 = conv3d_transpose(backward4, 3, 15, 15, scope='conv_2')
            backward6 = conv3d_transpose(backward5, 3, 15, 1, scope='conv_1', stride=2)

            output = backward6

        else:
            flattened = slim.flatten(forward6)
        
            self.flat = flattened

            with tf.variable_scope('fully_connected'):
                # fully connected layer
                output = slim.fully_connected(flattened, 2000, weights_regularizer=slim.l2_regularizer(self.config.l2))
                self.first_out = output
                output = slim.fully_connected(output, 500, weights_regularizer=slim.l2_regularizer(self.config.l2))
                output = slim.fully_connected(output, 2, activation_fn=None, weights_regularizer=slim.l2_regularizer(self.config.l2))

        return output


    def run_epoch(self, session, data, train_op=None):
        images, labels = data
        batch_size = self.config.batch_size
        total_steps = int(np.ceil(len(data[0])) / float(batch_size))
        total_loss = []
        accuracy = 0

        dp = self.config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1        

        # shuffle data
        p = np.random.permutation(images.shape[0])
        images, labels = images[p], labels[p]

        for step in xrange(total_steps):
            batch_start = step*batch_size
            index = range(batch_start,(batch_start + batch_size))
            feed = {self.f_images_placeholder: images[index],
                    self.autism_labels_placeholder: labels[index],
                    self.dropout_placeholder: dp}
            loss, pred, forward, _ = session.run([self.calculate_loss, self.pred, self.forward, train_op], feed_dict=feed)

            # save a sample of 10 image comparisons
            if step == 0 and self.config.mode == 'pretrain':
                _save_images(images[index], forward, 'features')

            answers = labels[index]
            accuracy += np.sum(pred == answers)/float(len(answers))
                    
            total_loss.append(loss)
            if step % 2 == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()

        sys.stdout.flush()
        sys.stdout.write('\n')

        return np.mean(total_loss), accuracy/float(total_steps)

        
    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholders()
        self.output = self.inference()
        self.pred = self.get_predictions(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_op(self.calculate_loss)




