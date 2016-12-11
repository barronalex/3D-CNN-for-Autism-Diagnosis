import tensorflow as tf
import numpy as np
import argparse
import time
import os

from cnn_3d import CNN_3D, Config
import mri_input
from test_cnn import test_cnn
import nn_utils

import sys

BATCH_SIZE = 15
MAX_STEPS = 10000
SAVE_EVERY = 100
MIN_EXAMPLES_IN_QUEUE = 1000
EARLY_STOPPING = 20

def train_cnn(config):

    mode = config.mode

    save_path = nn_utils.get_save_path(config)

    intro_str = '==> Building 3D CNN with %d layers'
    print intro_str % (config.num_layers)
    if config.use_sex_labels:
        print 'Debugging by training on gender labels'

    fn = 'data/mri_{}train.tfrecords'.format(config.gate)
    print '==> Reading examples from', fn
    filename_queue = tf.train.string_input_producer([fn], num_epochs=None)

    with tf.device('/cpu:0'):
        image, label, sex, corr = mri_input.read_and_decode_single_example(filename_queue,
                downsample_factor=config.downsample_factor, corr=config.use_correlation,
                rotate=config.rotate, noise=config.noise)

        image_batch, label_batch, sex_batch, corr_batch = tf.train.shuffle_batch(
            [image, label, sex, corr], batch_size=BATCH_SIZE,
            capacity=10000,
            min_after_dequeue=MIN_EXAMPLES_IN_QUEUE
            )

    label_batch = sex_batch if config.use_sex_labels else label_batch 

    cnn = CNN_3D(
            config,
            image_batch,
            label_batch,
            corr_batch,
            )

    # only restore layers that were previously trained
    pretrained_names = ['conv_' + str(i+1) + '/weights:0' for i in range(config.num_layers_to_restore)]
    pretrained_vars = [v for v in tf.all_variables() if v.name in pretrained_names]

    print '==> variables to be restored:'
    for v in pretrained_vars:
        print v.name
    print '==> variables to be trained:'
    for v in tf.trainable_variables():
        print v.name


    sess = tf.Session()

    summary_writer = tf.train.SummaryWriter('summaries/' + config.sum_dir + '/{}/train'.format(save_path[8:]), sess.graph)

    init = tf.initialize_all_variables()
    sess.run(init)


    if config.num_layers_to_restore > 0:
        restorer = tf.train.Saver(pretrained_vars)
        print '==> restoring weights'
        #path = 'weights/cae_pretrain_{}.weights'.format(str(num_layers_to_restore))
        path = 'weights/greedy_6layer_weights.weights'
        assert os.path.exists(path)
        restorer.restore(sess, path)
    if config.num_layers_to_restore == -1:
        restorer = tf.train.Saver()
        path = 'weights/cae_supervised.weights'
        assert os.path.exists(path)
        restorer.restore(sess, path)

    saver = tf.train.Saver()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_accuracy = 0
    best_val_loss = None

    for step in xrange(MAX_STEPS):
        start_time = time.time()
        _, loss_value, image_value, output_value, accuracy, summary = sess.run([
                    cnn.train_op,
                    cnn.loss,
                    image_batch,
                    cnn.outputs,
                    cnn.accuracy,
                    cnn.merged])
        #print output_value
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
                val_accuracy, val_loss = test_cnn(config, start_step=step)

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

                #cnn._save_images(image_value, filter_val, 'new_' + str(step))
            else:
                saver.save(sess, 
                        'weights/cae_' + mode + '_' + str(num_layers) + '.weights')
                #cnn._save_images(image_value, output_value, 'new_' + str(step))
    coord.request_stop()
    coord.join(threads)
    sess.close()
    tf.reset_default_graph()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="supervised")
    parser.add_argument("-l", "--num-layers", type=int, default=4)
    parser.add_argument("-r", "--num-layers-to-restore", type=int, default=0)
    parser.add_argument("-t", "--num-layers-to-train", type=int, default=4, 
            help="trains the specified number of innermost layers")
    parser.add_argument("-d", "--downsample_factor", type=int, default=2)
    parser.add_argument("-s", "--use_sex_labels", type=bool, default=False)
    parser.add_argument("-c", "--use_correlation", type=int, default=0, help="0 indicates no use, 1 supplements, 2 trains on only correlation")
    parser.add_argument('-g', '--gate', default='')
    parser.add_argument('-sd', '--sum-dir', default='')
    parser.add_argument('-ro', '--rotate', type=bool, default=True)
    parser.add_argument('-no', '--noise', type=float, default=0.1)
    args = parser.parse_args()
    config = Config()
    config.gate = args.gate
    config.num_layers = args.num_layers
    config.num_layers_to_train = args.num_layers_to_train
    config.mode = args.mode
    config.num_layers_to_restore = args.num_layers_to_restore
    config.downsample_factor = args.downsample_factor
    config.use_sex_labels = args.use_sex_labels
    config.use_correlation = args.use_correlation
    config.sum_dir = args.sum_dir
    config.rotate = args.rotate
    config.noise = args.noise
    train_cnn(config)

       
