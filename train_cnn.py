import tensorflow as tf
import argparse
import time

from cnn_3d import CNN_3D, Config

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default="pretrain")
parser.add_argument("-r", "--restore", default=False)
args = parser.parse_args()

config = Config()

config.mode = args.mode

with tf.variable_scope('3d_CAE'):
    model = CNN_3D(config)

pretrained_vars = [v for v in tf.trainable_variables() if 'fully_connected' not in v.name]

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

    if model.config.mode == 'supervised' or args.restore:
        print '==> restoring weights'
        saver.restore(session, 'weights/cae.weights')

    print '==> starting training'
    for epoch in xrange(model.config.max_epochs):
        print 'Epoch {}'.format(epoch)
        start = time.time()

        train_loss, train_accuracy = model.run_epoch(
          session, model.train,
          train_op=model.train_step)
        print 'Training loss: {}'.format(train_loss)
        if model.config.mode == 'supervised':
            valid_loss, valid_accuracy = model.run_epoch(session, model.val)
            print 'Validation loss: {}'.format(valid_loss)
            print 'Training accuracy: {}'.format(train_accuracy)
            print 'Vaildation accuracy: {}'.format(valid_accuracy)

        saver.save(session, 'weights/cae.weights')



        if model.config.mode == 'supervised':
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                #if best_val_loss < best_overall_val_loss:
                    #print 'Saving weights'
                    #best_overall_val_loss = best_val_loss
                #    best_val_accuracy = valid_accuracy
                ##saver.save(session, 'weights/task' + str(model.config.babi_id) + '.weights')

        # anneal
        #if train_loss>prev_epoch_loss*model.config.anneal_threshold:
            #model.config.lr/=model.config.anneal_by
            #print 'annealed lr to %f'%model.config.lr

        #prev_epoch_loss = train_loss


        #if epoch - best_val_epoch > model.config.early_stopping:
            #break
        print 'Total time: {}'.format(time.time() - start)

    #print 'Best validation accuracy:', best_val_accuracy
