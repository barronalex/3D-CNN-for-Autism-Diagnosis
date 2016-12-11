import tensorflow as tf
import numpy as np
import h5py
slim = tf.contrib.slim

import sys

DATA_DIR = 'data/eeg'
BATCH_SIZE = 32

def load_data(split='train'):
    data_file = h5py.File(DATA_DIR + '/{}.h5'.format(split), 'r')
    eeg = data_file['eeg'][:]
    lengths = data_file['lengths'][:]
    labels = data_file['labels'][:]

    print eeg.shape
    
    return eeg, lengths, labels


eeg_placeholder = tf.placeholder(tf.float32, shape=(None, 116, 316)) 
len_placeholder = tf.placeholder(tf.int32, shape=(None,)) 
labels_placeholder = tf.placeholder(tf.int64, shape=(None,)) 

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(200)

lstm_outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, eeg_placeholder, sequence_length=len_placeholder, dtype=tf.float32)

print lstm_outputs.get_shape()
print final_state.h.get_shape()

outputs = slim.fully_connected(final_state.h, 2, activation_fn=None)

print outputs.get_shape()

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, labels_placeholder)

correct_prediction = tf.equal(tf.argmax(outputs, 1), labels_placeholder)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

def run_epoch(session, data, verbose=2, train=True):

    total_steps = len(data[0]) / BATCH_SIZE
    total_loss = []
    accuracies = []
    
    # shuffle data
    p = np.random.permutation(len(data[0]))
    eeg, lengths, labels = data
    eeg, lengths, labels = eeg[p], lengths[p], labels[p] 


    for step in range(total_steps):
        index = range(step*BATCH_SIZE,(step+1)*BATCH_SIZE)
        feed = {eeg_placeholder: eeg[index],
                len_placeholder: lengths[index],
                labels_placeholder: labels[index]}

        train_operation = train_op if train else tf.no_op()
        loss_value, accuracy_value, _ = session.run(
          [loss, accuracy, train_operation], feed_dict=feed)

        #if train_writer is not None:
            #train_writer.add_summary(summary, num_epoch*total_steps + step)

        accuracies.append(accuracy_value) 


        total_loss.append(loss_value)
        if verbose and step % verbose == 0:
            sys.stdout.write('\r{} / {} : loss = {}'.format(
              step, total_steps, np.mean(total_loss)))
            sys.stdout.flush()

    if verbose:
        sys.stdout.write('\r')

    return np.mean(np.array(accuracies))

session = tf.Session()

init = tf.initialize_all_variables()
session.run(init)

train_data = load_data()
val_data = load_data(split='val')

for i in range(1000):
    train_accuracy = run_epoch(session, train_data)
    print ''
    print 'train accuracy:', train_accuracy
    val_accuracy = run_epoch(session, val_data, train=False)
    print ''
    print 'val accuracy:', val_accuracy



    



