import tensorflow as tf
import numpy as np
import h5py

import sys
import time

image_dimensions = [91, 109, 91]
DATA_DIR = 'data'


class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 10
    lr = 0.001
    hidden_size = 100
    max_epochs = 50
    early_stopping = 20


def _load_from_h5(filename):
    f = h5py.File(filename, 'r')
    data = f['data'][:]
    f.close()
    return data

#def conv3d(input, layer_name, kernel_size, num_filters, num_channels, strides=1, l2=0.0001):
    #weights = tf.get_variable('w_' + layer_name,
            #([kernel_size, kernel_size, kernel_size, num_filters, num_channels]))
    #outpu


class CNN_3D(object):
    
    def load_data(self):
        print '<== loading train/val data'
        train_images = _load_from_h5(DATA_DIR + '/train_images.h5')
        val_images = _load_from_h5(DATA_DIR + '/val_images.h5')

        train_labels = _load_from_h5(DATA_DIR + '/train_labels.h5')
        val_labels = _load_from_h5(DATA_DIR + '/val_labels.h5')

        self.train = train_images, train_labels
        self.val = val_images, val_labels

    def add_placeholders(self):
        self.f_images_placeholder = tf.placeholder(tf.float32, shape=[self.config.batch_size] + image_dimensions)
        self.autism_labels_placeholder = tf.placeholder(tf.int64, shape=(self.config.batch_size,))

    def get_predictions(self, output):
        """Get answer predictions from output"""
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred

    def add_loss_op(self, output):
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output, self.autism_labels_placeholder))

        return loss

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)

        return train_op

    def inference(self):
        images = tf.expand_dims(self.f_images_placeholder, -1)

        with tf.variable_scope('conv3d', initializer=_xavier_weight_init()):
            w1 = tf.get_variable('w1', (5,5,5,1,30))
            w2 = tf.get_variable('w2', (5,5,5,30,40))
            w3 = tf.get_variable('w3', (5,5,5,40,40))
            w4 = tf.get_variable('w4', (5,5,5,40,50))

            wf1 = tf.get_variable('wf1', (1800, 100))
            wf2 = tf.get_variable('wf2', (100, 2))
            b1 = tf.get_variable('b1', (100,))
            b2 = tf.get_variable('b2', (2,))

        layer1 = tf.nn.conv3d(images, w1, [1,2,2,2,1], 'SAME')
        print layer1.get_shape()
        layer1 = tf.nn.relu(layer1)
        layer2 = tf.nn.conv3d(layer1, w2, [1,2,2,2,1], 'SAME')
        print layer2.get_shape()
        layer2 = tf.nn.relu(layer2)
        layer3 = tf.nn.conv3d(layer2, w3, [1,2,2,2,1], 'SAME')
        print layer3.get_shape()
        layer3 = tf.nn.relu(layer3)
        layer4 = tf.nn.conv3d(layer3, w4, [1,2,2,2,1], 'SAME')
        print layer4.get_shape()
        layer4 = tf.nn.relu(layer4)
        max_pool = tf.nn.max_pool3d(layer4, [1,3,3,3,1], [1,2,2,2,1], 'SAME')
        print max_pool.get_shape()
        flattened = tf.reshape(max_pool, [self.config.batch_size, -1])
        print flattened.get_shape()

        # fully connected layer
        output = tf.matmul(tf.nn.tanh(tf.matmul(flattened, wf1) + b1), wf2) + b2

        print output.get_shape()
        return output



    def run_epoch(self, session, data, train_op=None):
        images, labels = data
        batch_size = self.config.batch_size
        total_steps = int(np.ceil(len(data[0])) / float(batch_size))
        total_loss = []
        accuracy = 0

        if train_op is None:
            train_op = tf.no_op()

        for step in xrange(total_steps):
            batch_start = step*batch_size
            index = range(batch_start,(batch_start + batch_size))
            feed = {self.f_images_placeholder: images[index],
                    self.autism_labels_placeholder: labels[index]}
            loss, pred, _ = session.run([self.calculate_loss, self.pred, train_op], feed_dict=feed)

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

config = Config()
model = CNN_3D(config)

init = tf.initialize_all_variables()

session = tf.Session()
with session:
    print '==> initializing variables'
    session.run(init)
    
    best_val_epoch = 0
    prev_epoch_loss = float('inf')
    best_val_loss = float('inf')
    best_val_accuracy = 0.0

    #if args.restore:
        #print '==> restoring weights'
        #saver.restore(session, 'weights/task' + str(model.config.babi_id) + '.weights')

    print '==> starting training'
    for epoch in xrange(model.config.max_epochs):
        print 'Epoch {}'.format(epoch)
        start = time.time()

        train_loss, train_accuracy = model.run_epoch(
          session, model.train,
          train_op=model.train_step)
        valid_loss, valid_accuracy = model.run_epoch(session, model.val)
        print 'Training loss: {}'.format(train_loss)
        print 'Validation loss: {}'.format(valid_loss)
        print 'Training accuracy: {}'.format(train_accuracy)
        print 'Vaildation accuracy: {}'.format(valid_accuracy)

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_val_epoch = epoch
            #if best_val_loss < best_overall_val_loss:
                ##print 'Saving weights'
                #best_overall_val_loss = best_val_loss
                #best_val_accuracy = valid_accuracy
                ##saver.save(session, 'weights/task' + str(model.config.babi_id) + '.weights')

        # anneal
        #if train_loss>prev_epoch_loss*model.config.anneal_threshold:
            #model.config.lr/=model.config.anneal_by
            #print 'annealed lr to %f'%model.config.lr

        prev_epoch_loss = train_loss


        if epoch - best_val_epoch > model.config.early_stopping:
            break
        print 'Total time: {}'.format(time.time() - start)

    print 'Best validation accuracy:', best_val_accuracy



