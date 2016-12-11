import tensorflow as tf
import numpy as np

import os
import argparse

from cnn_3d import CNN_3D, Config
import mri_input
import nn_utils

BATCH_SIZE = 15
NUM_EXAMPLES = {'train': 749, 'val': 107, 'test': 215}

# only need to test the cnn when 
def test_cnn(config, dataset='val', start_step=0):

    test_graph = tf.Graph()

    best = True if dataset == 'test' else False

    restore_path = nn_utils.get_save_path(config)

    if best:
        restore_path += '_best'

    with test_graph.as_default():

        fn = 'data/mri_{}.tfrecords'.format(config.gate + dataset)
        print fn
        filename_queue = tf.train.string_input_producer([fn], num_epochs=1)

        with tf.device('/cpu:0'):
            image, label, sex, corr = mri_input.read_and_decode_single_example(filename_queue, train=False,
                    downsample_factor=config.downsample_factor)

            image_batch, label_batch, sex_batch, corr_batch = tf.train.batch(
                [image, label, sex, corr], batch_size=BATCH_SIZE,
                capacity=100,
                allow_smaller_final_batch=True
                )

        label_batch = sex_batch if config.use_sex_labels else label_batch 

        cnn = CNN_3D(
                config,
                image_batch,
                label_batch,
                corr_batch,
                train=False
                )

        sess = tf.Session()

        summary_writer = tf.train.SummaryWriter('summaries/' + config.sum_dir + '/{}/test'.format(restore_path[8:]))

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset-split', default='val')
    parser.add_argument("-m", "--mode", default="supervised")
    parser.add_argument("-l", "--num-layers", type=int, default=4)
    parser.add_argument("-r", "--num-layers-to-restore", type=int, default=0)
    parser.add_argument("-t", "--num-layers-to-train", type=int, default=4, 
            help="trains the specified number of innermost layers")
    parser.add_argument("-d", "--downsample_factor", type=int, default=2)
    parser.add_argument("-s", "--use_sex_labels", type=bool, default=False)
    parser.add_argument("-c", "--use_correlation", type=int, default=0, help="0 indicates no use, 1 supplements, 2 trains on only correlation")
    parser.add_argument('-g', '--gate', default='')
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
    accuracy, loss = test_cnn(config, dataset=args.dataset_split)
    print args.dataset_split, 'accuracy:', accuracy
    print args.dataset_split, 'loss:', loss



