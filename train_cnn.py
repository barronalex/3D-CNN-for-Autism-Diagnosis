import tensorflow as tf
import argparse
import time

from cnn_3d import CNN_3D, Config

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default="pretrain")
parser.add_argument("-r", "--restore", default=1)
args = parser.parse_args()

config = Config()

config.mode = args.mode

with tf.variable_scope('3d_CAE'):
    model = CNN_3D(config)

# I'm setting certain variables to be not trained? Then trying to restore them>
pretrained_vars = [v for v in tf.all_variables() if 'fully_connected' not in v.name and 'Adam' not in v.name and 'mean' not in v.name and 'variance' not in v.name and 'power' not in v.name]

#print 'saved vars'
#for v in pretrained_vars:
    #print v.name

#print 'other vars'
#for v in tf.trainable_variables():
    #print v.name

saver = tf.train.Saver(pretrained_vars)

init = tf.initialize_all_variables()


session = tf.Session()
with session:
    print '==> initializing variables'
    session.run(init)
    
    best_val_epoch = 0
    prev_epoch_loss = float('inf')
    best_val_loss = float('inf')
    best_val_accuracy = 0.0

    if int(args.restore):
        print '==> restoring weights'
        saver.restore(session, 'weights/cae_pretrain.weights')

    print '==> starting training'
    for epoch in xrange(model.config.max_epochs):
        print 'Epoch {}'.format(epoch)
        start = time.time()

        train_loss, train_accuracy = model.run_epoch(
          session, model.train,
          train_op=model.train_step)
        print 'Training loss: {}'.format(train_loss)
        #if model.config.mode == 'supervised':
        valid_loss, valid_accuracy = model.run_epoch(session, model.val)
        print 'Validation loss: {}'.format(valid_loss)
        print 'Training accuracy: {}'.format(train_accuracy)
        print 'Vaildation accuracy: {}'.format(valid_accuracy)

        saver.save(session, 'weights/cae_' + model.config.mode + '.weights')

        if model.config.mode == 'supervised':
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch

            if epoch - best_val_epoch > model.config.early_stopping:
                break
        print 'Total time: {}'.format(time.time() - start)

