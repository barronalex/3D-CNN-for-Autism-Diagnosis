import tensorflow as tf
slim = tf.contrib.slim

def conv3d(inputs, kernel_size, num_channels, num_filters, scope='', stride=1, activation=tf.nn.relu, l2=0.0001, padding='SAME', trainable=True):

    with tf.variable_scope(scope, initializer = slim.xavier_initializer()):

        weights = tf.get_variable('weights',
                ([kernel_size, kernel_size, kernel_size, num_channels, num_filters]), trainable=trainable)
        output = tf.nn.conv3d(inputs, weights, [1, stride, stride, stride, 1], padding)

    output = slim.bias_add(output, reuse=False)
    output = activation(output)
    output = slim.batch_norm(output)

    # add l2 reg
    reg = l2*tf.nn.l2_loss(weights)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)
    
    return output

def conv3d_transpose(inputs, kernel_size, num_channels, num_filters, scope='', stride=1, activation=tf.nn.relu, l2=0.0001, padding='SAME', trainable=True):

    def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
        dim_size *= stride_size
        if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
        return dim_size

    with tf.variable_scope(scope, initializer = slim.xavier_initializer(), reuse=True):

        weights = tf.get_variable('weights',
                ([kernel_size, kernel_size, kernel_size, num_filters, num_channels]))

        batch_size, height, width, depth, _ = inputs.get_shape()

        out_height = get_deconv_dim(height, stride, kernel_size, padding)
        out_width = get_deconv_dim(width, stride, kernel_size, padding)
        out_depth = get_deconv_dim(depth, stride, kernel_size, padding)

        output_shape = tf.pack([tf.shape(inputs)[0], int(out_height), int(out_width), int(out_depth), int(num_filters)])

        output = tf.nn.conv3d_transpose(inputs, weights, output_shape, [1, stride, stride, stride, 1], padding=padding)
        
        out_shape = inputs.get_shape().as_list()
        out_shape[-1] = num_filters
        out_shape[1] = get_deconv_dim(out_shape[1], stride, kernel_size, padding)
        out_shape[2] = get_deconv_dim(out_shape[2], stride, kernel_size, padding)
        out_shape[3] = get_deconv_dim(out_shape[3], stride, kernel_size, padding)
        output.set_shape(out_shape)

    output = slim.bias_add(output, reuse=False)
    output = activation(output)
    output = slim.batch_norm(output)

    # add l2 reg
    reg = l2*tf.nn.l2_loss(weights)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)
    
    return output
