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

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default="pretrain")
parser.add_argument("-r", "--restore", default=1)
args = parser.parse_args()

fn = 'data/mri_train.tfrecords'
filename_queue = tf.train.string_input_producer([fn], num_epochs=None)
image, label = mri_input.read_and_decode_single_example(filename_queue)

image_batch, label_batch = tf.train.shuffle_batch(
    [image, label], batch_size=BATCH_SIZE,
    capacity=100,
    min_after_dequeue=50
    )

# train as auto-encoder in pretraining
label_batch = image_batch if args.mode == 'pretrain' else label_batch

outputs = cnn_3d.inference(image_batch, args.mode)

loss = cnn_3d.loss(outputs, label_batch, args.mode)

train_op = cnn_3d.train_op(loss)

pretrained_vars = [v for v in tf.all_variables() if 'fully_connected' not in v.name and 'Adam' not in v.name and 'mean' not in v.name and 'variance' not in v.name and 'power' not in v.name]

saver = tf.train.Saver(pretrained_vars)

sess = tf.Session()

init = tf.initialize_all_variables()
sess.run(init)

if int(args.restore):
    print '==> restoring weights'
    if os.path.exists('weights/cae_pretrain.weights'):
        saver.restore(sess, 'weights/cae_pretrain.weights')

tf.train.start_queue_runners(sess=sess)

for step in xrange(MAX_STEPS):
    start_time = time.time()
    _, loss_value, ib, o = sess.run([train_op, loss, image_batch, outputs])
    duration = time.time() - start_time

    if step % 10 == 0:
        if args.mode == 'pretrain':
            cnn_3d._save_images(ib, o, 'new_' + str(step) + '_')
        num_examples_per_step = BATCH_SIZE
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        s = format_str % (step, loss_value,
                             examples_per_sec, sec_per_batch)
        sys.stdout.write('\r' + s + ' ')
        sys.stdout.flush()

    if (step % 100 == 0 or (step + 1) == MAX_STEPS):
         saver.save(sess, 'weights/cae_' + args.mode + '.weights')
         test_cnn(mode=args.mode)
         


