import tensorflow as tf
import numpy as np

import os
import argparse

import cnn_3d
import mri_input

BATCH_SIZE = 15
NUM_EXAMPLES = {'train': 749, 'val': 107, 'test': 215}

# only need to test the cnn when 
def test_cnn(num_layers, dataset='val', mode='supervised'):

    test_graph = tf.Graph()

    with test_graph.as_default():

        fn = 'data/mri_{}.tfrecords'.format(dataset)
        filename_queue = tf.train.string_input_producer([fn], num_epochs=1)

        with tf.device('/cpu:0'):
            image, label = mri_input.read_and_decode_single_example(filename_queue, train=False)

        image_batch, label_batch = tf.train.batch(
            [image, label], batch_size=BATCH_SIZE,
            capacity=100,
            allow_smaller_final_batch=True
            )

        # train as auto-encoder in pretraining
        label_batch = image_batch if mode == 'pretrain' else label_batch

        outputs, filt = cnn_3d.inference(image_batch, num_layers, 0, mode, False)

        loss = cnn_3d.loss(outputs, label_batch, mode)

        predictions = cnn_3d.predictions(outputs)

        restorer = tf.train.Saver()

        sess = tf.Session()

        init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess.run(init)

        if os.path.exists('weights/cae_pretrain.weights'):
            restorer.restore(sess, 'weights/cae_{}.weights'.format(mode))

        tf.train.start_queue_runners(sess=sess)

        step = 0
        accuracy = 0
        overall_loss = 0
        try:
            while True:
                loss_value, pred_value, labels_value, output_val = sess.run([loss, predictions, label_batch, outputs])
                #print np.sum(labels_value)
                accuracy += np.sum(pred_value == labels_value)/float(pred_value.shape[0])
                overall_loss += loss_value
                #print np.sum(pred_value == labels_value)/float(pred_value.shape[0])

                step += 1
        except tf.errors.OutOfRangeError:
            return accuracy/float(step), overall_loss/float(step)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='val')
    parser.add_argument('-l', '--num-layers', default=6)
    args = parser.parse_args()
    test_cnn(args.num_layers, dataset=args.dataset)


