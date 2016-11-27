import tensorflow as tf
import numpy as np
import argparse
import time
import os

import cnn_3d
import mri_input
from test_cnn import *

import sys

BATCH_SIZE = 15
MAX_STEPS = 10000
SAVE_EVERY = 100
MIN_IMAGES_IN_QUEUE = 1000

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default="pretrain")
parser.add_argument("-l", "--num-layers", type=int, default=4)
parser.add_argument("-r", "--num-layers-to-restore", type=int, default=-1)
parser.add_argument("-t", "--num-layers-to-train", type=int, default=-1, 
        help="trains the specified number of innermost layers")
parser.add_argument("-d", "--downsample_factor", type=int, default=1)
args = parser.parse_args()

if args.num_layers_to_restore == -1:
    args.num_layers_to_restore = args.num_layers

if args.num_layers_to_train == -1:
    args.num_layers_to_train = args.num_layers - args.num_layers_to_restore


def train_cnn(mode='pretrain', num_layers=2, num_layers_to_restore=2,
        num_layers_to_train=2, downsample_factor=1):
    fn = 'data/mri_train.tfrecords'
    filename_queue = tf.train.string_input_producer([fn], num_epochs=None)
    image, label = mri_input.read_and_decode_single_example(filename_queue, downsample_factor=1)

    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=BATCH_SIZE,
        capacity=10000,
        min_after_dequeue=MIN_IMAGES_IN_QUEUE
        )

    # train as auto-encoder in pretraining
    label_batch = image_batch if mode == 'pretrain' else label_batch

    outputs = cnn_3d.inference(image_batch, num_layers, num_layers_to_train, mode)

    loss = cnn_3d.loss(outputs, label_batch, mode)

    predictions = cnn_3d.predictions(outputs)

    train_op = cnn_3d.train_op(loss)

    # only restore layers that were previously trained
    pretrained_names = ['conv_' + str(i+1) + '/weights:0' for i in range(num_layers_to_restore)]
    pretrained_vars = [v for v in tf.all_variables() if v.name in pretrained_names]

    print '==> variables to be restored:'
    for v in pretrained_vars:
        print v.name
    print '==> variables to be trained:'
    for v in tf.all_variables():
        print v.name


    sess = tf.Session()

    init = tf.initialize_all_variables()
    sess.run(init)

    saver = tf.train.Saver()

    if num_layers_to_restore > 0:
        restorer = tf.train.Saver(pretrained_vars)
        print '==> restoring weights'
        path = 'weights/cae_pretrain_{}.weights'.format(str(num_layers_to_restore))
        assert os.path.exists(path)
        restorer.restore(sess, path)

    tf.train.start_queue_runners(sess=sess)

    train_accuracy = 0
    best_val_loss = None

    for step in xrange(MAX_STEPS):
        start_time = time.time()
        _, loss_value, pred_value, labels_value, image_value, output_value = sess.run(
                [train_op, loss, predictions, label_batch, image_batch, outputs])
        train_accuracy += np.sum(pred_value == labels_value)/float(pred_value.shape[0])
        duration = time.time() - start_time

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
                print '==> evaluating valid and train accuracy'
                val_accuracy, val_loss = test_cnn(num_layers, mode=mode)

                print 'train accuracy:', train_accuracy/float(SAVE_EVERY) 
                print 'val accuracy:', val_accuracy
                train_accuracy = 0

                if val_loss < best_val_loss or best_val_loss is None:
                    best_val_loss = val_loss
                    print '==> saving weights'
                    saver.save(sess, 
                            'weights/cae_' + mode + '.weights')

            else:
                cnn_3d._save_images(image_value, output_value, 'new_' + str(step) + '_')
                saver.save(sess, 
                        'weights/cae_' + mode + '_' + str(num_layers) + '.weights')
    sess.close()

if __name__ == '__main__':
    train_cnn(args.mode, args.num_layers, args.num_layers_to_restore, args.num_layers_to_train)
        
