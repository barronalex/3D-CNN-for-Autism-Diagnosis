import tensorflow as tf
import numpy as np
import argparse
import time
import os

from cnn_3d import CNN_3D
import mri_input
from test_cnn import *
import nn_utils

import sys

BATCH_SIZE = 15
MAX_STEPS = 100000
SAVE_EVERY = 250
MIN_IMAGES_IN_QUEUE = 1000
EARLY_STOPPING = 20

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default="supervised")
parser.add_argument("-l", "--num-layers", type=int, default=4)
parser.add_argument("-r", "--num-layers-to-restore", type=int, default=0)
parser.add_argument("-t", "--num-layers-to-train", type=int, default=-1, 
        help="trains the specified number of innermost layers")
parser.add_argument("-d", "--downsample_factor", type=int, default=1)
parser.add_argument("-s", "--use_sex_labels", type=bool, default=False)
args = parser.parse_args()



def train_cnn(mode='supervised', num_layers=2, num_layers_to_restore=2,
        num_layers_to_train=2, downsample_factor=1, use_sex_labels=False):

    params = [mode, num_layers, num_layers_to_train, downsample_factor, use_sex_labels]
    param_names = ['mode', 'num_layers', 'num_layers_to_train', 'downsample_factor', 'use_sex_labels']
    save_path = nn_utils.get_save_path(params, param_names)

    intro_str = '==> building 3D CNN with %d layers'
    print intro_str % (num_layers)
    if use_sex_labels:
        print 'Debugging by training on gender labels'

    fn = 'data/mri_train.tfrecords'
    filename_queue = tf.train.string_input_producer([fn], num_epochs=None)

    with tf.device('/cpu:0'):
        image, label, sex = mri_input.read_and_decode_single_example(filename_queue, downsample_factor=downsample_factor)

        image_batch, label_batch, sex_batch = tf.train.shuffle_batch(
            [image, label, sex], batch_size=BATCH_SIZE,
            capacity=10000,
            min_after_dequeue=MIN_IMAGES_IN_QUEUE
            )

    label_batch = sex_batch if use_sex_labels else label_batch 

    cnn = CNN_3D(image_batch, label_batch, num_layers, mode,
            num_layers_to_train)

    # only restore layers that were previously trained
    pretrained_names = ['conv_' + str(i+1) + '/weights:0' for i in range(num_layers_to_restore)]
    pretrained_vars = [v for v in tf.all_variables() if v.name in pretrained_names]

    print '==> variables to be restored:'
    for v in pretrained_vars:
        print v.name
    print '==> variables to be trained:'
    for v in tf.trainable_variables():
        print v.name


    sess = tf.Session()

    summary_writer = tf.train.SummaryWriter('summaries/{}/train'.format(save_path[8:]), sess.graph)

    init = tf.initialize_all_variables()
    sess.run(init)


    if num_layers_to_restore > 0:
        restorer = tf.train.Saver(pretrained_vars)
        print '==> restoring weights'
        path = 'weights/cae_pretrain_{}.weights'.format(str(num_layers_to_restore))
        assert os.path.exists(path)
        restorer.restore(sess, path)
    if num_layers_to_restore == -1:
        restorer = tf.train.Saver()
        path = 'weights/cae_supervised.weights'
        assert os.path.exists(path)
        restorer.restore(sess, path)

    saver = tf.train.Saver()

    tf.train.start_queue_runners(sess=sess)

    train_accuracy = 0
    best_val_loss = None

    for step in xrange(MAX_STEPS):
        start_time = time.time()
        _, loss_value, image_value, output_value, accuracy, filter_val, summary = sess.run([
                    cnn.train_op,
                    cnn.loss,
                    image_batch,
                    cnn.outputs,
                    cnn.accuracy,
                    cnn.filt,
                    cnn.merged])
        train_accuracy += accuracy
        duration = time.time() - start_time
        summary_writer.add_summary(summary, step)

        if step % 2 == 0:
            num_examples_per_step = BATCH_SIZE
            sec_per_batch = float(duration)

            format_str = ('step %s, loss = %.2f (%.3f '
                          'sec/batch)')
            s = format_str % (step, loss_value, sec_per_batch)
            sys.stdout.write('\r' + s + ' ')
            sys.stdout.flush()

        if (step % SAVE_EVERY == 0 or (step + 1) == MAX_STEPS) and step != 0:
            if mode == 'supervised':
                saver.save(sess, save_path)
                print '==> evaluating valid and train accuracy'
                val_accuracy, val_loss = test_cnn(mode, num_layers, num_layers_to_train,
                        downsample_factor, use_sex_labels, start_step=step)

                print 'train accuracy:', train_accuracy/float(SAVE_EVERY) 
                print 'val accuracy:', val_accuracy
                train_accuracy = 0

                if val_loss < best_val_loss or best_val_loss is None:
                    early_stopping_count = 0
                    best_val_loss = val_loss
                    print '==> saving weights to', save_path + '_best'
                    saver.save(sess, save_path + '_best')
                else:
                    early_stopping_count += 1
                    if early_stopping_count >= EARLY_STOPPING: break

                cnn._save_images(image_value, filter_val, 'new_' + str(step))
            else:
                saver.save(sess, 
                        'weights/cae_' + mode + '_' + str(num_layers) + '.weights')
                cnn._save_images(image_value, output_value, 'new_' + str(step))
    sess.close()

if __name__ == '__main__':
    train_cnn(args.mode, args.num_layers, args.num_layers_to_restore,
            args.num_layers_to_train, args.downsample_factor, args.use_sex_labels)
       
