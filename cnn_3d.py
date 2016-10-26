import tensorflow as tf
import numpy as np
import h5py
import matplotlib.image as mpimg

import sys
import time

slim = tf.contrib.slim

image_dimensions = [96, 112, 96]
DATA_DIR = 'data'


class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 5
    lr = 0.001
    hidden_size = 100
    max_epochs = 100
    early_stopping = 20

    mode = 'pretrain'


def _load_from_h5(filename):
    f = h5py.File(filename, 'r')
    data = f['data'][:]
    f.close()
    return data

def conv3d(inputs, kernel_size, num_channels, num_filters, scope='', stride=1, activation=tf.nn.relu, l2=0.0001, padding='SAME', transpose=False, trainable=True):


    def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
        dim_size *= stride_size
        if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
        return dim_size

    if transpose:

        with tf.variable_scope(scope, initializer = slim.xavier_initializer(), reuse=True):

            weights = tf.get_variable('weights',
                    ([kernel_size, kernel_size, kernel_size, num_filters, num_channels]))

            batch_size, height, width, depth, _ = inputs.get_shape()

            out_height = get_deconv_dim(height, stride, kernel_size, padding)
            out_width = get_deconv_dim(width, stride, kernel_size, padding)
            out_depth = get_deconv_dim(depth, stride, kernel_size, padding)

            output_shape = int(batch_size), int(out_height), int(out_width), int(out_depth), int(num_filters)

            output = tf.nn.conv3d_transpose(inputs, weights, output_shape, [1, stride, stride, stride, 1], padding=padding)
            
            out_shape = inputs.get_shape().as_list()
            out_shape[-1] = num_filters
            out_shape[1] = get_deconv_dim(out_shape[1], stride, kernel_size, padding)
            out_shape[2] = get_deconv_dim(out_shape[2], stride, kernel_size, padding)
            out_shape[3] = get_deconv_dim(out_shape[3], stride, kernel_size, padding)
            output.set_shape(out_shape)
        
    else:

        with tf.variable_scope(scope, initializer = slim.xavier_initializer()):

            weights = tf.get_variable('weights',
                    ([kernel_size, kernel_size, kernel_size, num_channels, num_filters]), trainable=trainable)
            output = tf.nn.conv3d(inputs, weights, [1, stride, stride, stride, 1], padding)

    output = slim.bias_add(output, reuse=False)

    output = activation(output)

    print output.get_shape()

    # add l2 reg
    reg = l2*tf.nn.l2_loss(weights)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)
    
    return output

def _save_first_image(images):
    mpimg.imsave("output.png", images[0,35,:,:,0])

class CNN_3D(object):
    
    def load_data(self):
        print '<== loading train/val data'
        train_images = _load_from_h5(DATA_DIR + '/train_images.h5')
        val_images = _load_from_h5(DATA_DIR + '/val_images.h5')

        train_images = np.expand_dims(train_images, axis=-1)
        val_images = np.expand_dims(val_images, axis=-1)

        mpimg.imsave('actual.png', train_images[0,35,:,:,0])

        train_labels = _load_from_h5(DATA_DIR + '/train_labels.h5')
        val_labels = _load_from_h5(DATA_DIR + '/val_labels.h5')

        self.train = train_images, train_labels
        self.val = val_images, val_labels

    def add_placeholders(self):
        self.f_images_placeholder = tf.placeholder(tf.float32, shape=[self.config.batch_size] + image_dimensions + [1])
        self.autism_labels_placeholder = tf.placeholder(tf.int64, shape=(self.config.batch_size,))

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

        return loss

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)

        return train_op

    def inference(self):

        trainable = False

        forward1 = conv3d(self.f_images_placeholder, 3, 1, 30, scope='conv_1', trainable=trainable)
        forward2 = conv3d(forward1, 3, 30, 30, scope='conv_2', stride=2, trainable=trainable)

        forward3 = conv3d(forward2, 3, 30, 30, scope='conv_3', trainable=trainable)
        forward4 = conv3d(forward3, 3, 30, 30, scope='conv_4', stride=2, trainable=trainable)

        forward5 = conv3d(forward4, 3, 30, 30, scope='conv_5', trainable=trainable)
        forward6 = conv3d(forward5, 3, 30, 30, scope='conv_6', stride=2, trainable=trainable)
        self.forward = forward4

        print forward6.get_shape()

        if self.config.mode == 'pretrain':
            backward1 = conv3d(forward6, 3, 30, 30, scope='conv_6', transpose=True)
            backward2 = conv3d(backward1, 3, 30, 30, scope='conv_5', stride=2, transpose=True)

            backward3 = conv3d(forward4, 3, 30, 30, scope='conv_4', transpose=True)
            backward4 = conv3d(backward3, 3, 30, 30, scope='conv_3', stride=2, transpose=True)

            backward5 = conv3d(backward4, 3, 30, 30, scope='conv_2', transpose=True)
            backward6 = conv3d(backward5, 3, 30, 1, scope='conv_1', stride=2, transpose=True)

            output = backward6

        else:
            flattened = tf.reshape(forward4, [self.config.batch_size, -1])

            print flattened.get_shape()

            with tf.variable_scope('fully_connected'):
                # fully connected layer
                output = slim.fully_connected(flattened, 2000)
                self.first_out = output
                output = slim.fully_connected(output, 500)
                output = slim.fully_connected(output, 2)

        return output


    def run_epoch(self, session, data, train_op=None):
        images, labels = data
        batch_size = self.config.batch_size
        total_steps = int(np.ceil(len(data[0])) / float(batch_size))
        total_loss = []
        accuracy = 0

        # shuffle data
        p = np.random.permutation(images.shape[0])
        images, labels = images[p], labels[p]

        if train_op is None:
            train_op = tf.no_op()

        for step in xrange(total_steps):
            batch_start = step*batch_size
            index = range(batch_start,(batch_start + batch_size))
            feed = {self.f_images_placeholder: images[index],
                    self.autism_labels_placeholder: labels[index]}
            loss, pred, output, _ = session.run([self.calculate_loss, self.pred, self.forward, train_op], feed_dict=feed)

            #print output

            if step == 0 and self.config.mode == 'pretrain':
                #_save_first_image(output)
                pass

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




