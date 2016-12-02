import tensorflow as tf
import numpy as np

import os
import argparse

from cnn_3d import CNN_3D
import mri_input
import nn_utils

BATCH_SIZE = 15
NUM_EXAMPLES = {'train': 749, 'val': 107, 'test': 215}

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', default='supervised')
parser.add_argument('-s', '--dataset-split', default='val')
parser.add_argument('-l', '--num-layers', type=int, default=6)
parser.add_argument('-t', '--num_layers_to_train', type=int, default=-1) 
parser.add_argument('-d', '--downsample_factor', type=int, default=1)
args = parser.parse_args()

# only need to test the cnn when 
def test_cnn(mode='supervised', num_layers=2,
        num_layers_to_train=2, downsample_factor=1,
        use_sex_labels=False, dataset='val', start_step=0):

    test_graph = tf.Graph()

    best = True if dataset == 'test' else False

    params = [mode, num_layers, num_layers_to_train, downsample_factor, use_sex_labels]
    param_names = ['mode', 'num_layers', 'num_layers_to_train', 'downsample_factor', 'use_sex_labels']
    restore_path = nn_utils.get_save_path(params, param_names)
    if best:
        restore_path += '_best'

    with test_graph.as_default():

        fn = 'data/mri_{}.tfrecords'.format(dataset)
        filename_queue = tf.train.string_input_producer([fn], num_epochs=1)

        with tf.device('/cpu:0'):
            image, label, sex, corr = mri_input.read_and_decode_single_example(filename_queue, train=False,
                    downsample_factor=downsample_factor)

            image_batch, label_batch, sex_batch, corr_batch = tf.train.batch(
                [image, label, sex, corr], batch_size=BATCH_SIZE,
                capacity=100,
                allow_smaller_final_batch=True
                )

        label_batch = sex_batch if use_sex_labels else label_batch 

        cnn = CNN_3D(
                image_batch,
                label_batch,
                corr_batch,
                num_layers,
                mode,
                train=False
                )

        sess = tf.Session()

        summary_writer = tf.train.SummaryWriter('summaries/{}/test'.format(restore_path[8:]))

        init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess.run(init)

        restorer = tf.train.Saver()

        assert os.path.exists(restore_path)
        print 'restoring from', restore_path
        restorer.restore(sess, restore_path)

        tf.train.start_queue_runners(sess=sess)

        step = 0
        val_accuracy = 0
        overall_loss = 0
        try:
            while True:
                loss_value, accuracy, output_val, summary = sess.run([
                    cnn.loss,
                    cnn.accuracy,
                    cnn.outputs,
                    cnn.merged])
                val_accuracy += accuracy
                overall_loss += loss_value
                summary_writer.add_summary(summary, start_step + step)
                step += 1
        except tf.errors.OutOfRangeError:
            return val_accuracy/float(step), overall_loss/float(step)
        
if __name__ == '__main__':
    accuracy, loss = test_cnn(args.mode, args.num_layers, args.num_layers_to_train,
            args.downsample_factor, dataset=args.dataset_split)
    print args.dataset_split, 'accuracy:', accuracy
    print args.dataset_split, 'loss:', loss


