import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

def conv3d(inputs, kernel_size, num_channels, num_filters,
        scope='', stride=1, activation=tf.nn.relu, l2=0.0, padding='SAME', trainable=True):

    with tf.variable_scope(scope, initializer = slim.xavier_initializer()):

        weights = tf.get_variable('weights',
                ([kernel_size, kernel_size, kernel_size, num_channels, num_filters]), trainable=trainable)
        output = tf.nn.conv3d(inputs, weights, [1, stride, stride, stride, 1], padding)

    #output = slim.bias_add(output, reuse=False)
    output = activation(output)
    output = slim.batch_norm(output)

    # add l2 reg
    reg = l2*tf.nn.l2_loss(weights)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)
    
    return output

def conv3d_transpose(inputs, kernel_size, num_channels, num_filters,
        scope='', stride=1, activation=tf.nn.relu, l2=0.0, padding='SAME'):

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

    #output = slim.bias_add(output, reuse=False)
    output = activation(output)
    output = slim.batch_norm(output)

    # add l2 reg
    reg = l2*tf.nn.l2_loss(weights)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)
    
    return output

# from http://stackoverflow.com/questions/34801342/tensorflow-how-to-rotate-an-image-for-data-augmentation/40483687#40483687
def rotate_image_tensor(image, angle, mode='black'):
    """
    Rotates a 3D tensor (HWD), which represents an image by given radian angle.

    New image has the same size as the input image.

    mode controls what happens to border pixels.
    mode = 'black' results in black bars (value 0 in unknown areas)
    mode = 'white' results in value 255 in unknown areas
    mode = 'ones' results in value 1 in unknown areas
    mode = 'repeat' keeps repeating the closest pixel known
    """
    # likely a problem with the shape of the image
    s = image.get_shape().as_list()
    assert len(s) == 3, "Input needs to be 3D."
    assert (mode == 'repeat') or (mode == 'black') or (mode == 'white') or (mode == 'ones'), "Unknown boundary mode."
    image_center = [np.floor(x/2) for x in s]

    # Coordinates of new image
    coord1 = tf.range(s[0])
    coord2 = tf.range(s[1])

    # Create vectors of those coordinates in order to vectorize the image
    coord1_vec = tf.tile(coord1, [s[1]])

    coord2_vec_unordered = tf.tile(coord2, [s[0]])
    coord2_vec_unordered = tf.reshape(coord2_vec_unordered, [s[0], s[1]])
    coord2_vec = tf.reshape(tf.transpose(coord2_vec_unordered, [1, 0]), [-1])

    # center coordinates since rotation center is supposed to be in the image center
    coord1_vec_centered = coord1_vec - image_center[0]
    coord2_vec_centered = coord2_vec - image_center[1]

    coord_new_centered = tf.cast(tf.pack([coord1_vec_centered, coord2_vec_centered]), tf.float32)

    # Perform backward transformation of the image coordinates
    rot_mat_inv = tf.dynamic_stitch([0, 1, 2, 3], [tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)])
    rot_mat_inv = tf.reshape(rot_mat_inv, shape=[2, 2])
    coord_old_centered = tf.matmul(rot_mat_inv, coord_new_centered)

    # Find nearest neighbor in old image
    coord1_old_nn = tf.cast(tf.round(coord_old_centered[0, :] + image_center[0]), tf.int32)
    coord2_old_nn = tf.cast(tf.round(coord_old_centered[1, :] + image_center[1]), tf.int32)

    # Clip values to stay inside image coordinates
    if mode == 'repeat':
        coord_old1_clipped = tf.minimum(tf.maximum(coord1_old_nn, 0), s[0]-1)
        coord_old2_clipped = tf.minimum(tf.maximum(coord2_old_nn, 0), s[1]-1)
    else:
        outside_ind1 = tf.logical_or(tf.greater(coord1_old_nn, s[0]-1), tf.less(coord1_old_nn, 0))
        outside_ind2 = tf.logical_or(tf.greater(coord2_old_nn, s[1]-1), tf.less(coord2_old_nn, 0))
        outside_ind = tf.logical_or(outside_ind1, outside_ind2)

        coord_old1_clipped = tf.boolean_mask(coord1_old_nn, tf.logical_not(outside_ind))
        coord_old2_clipped = tf.boolean_mask(coord2_old_nn, tf.logical_not(outside_ind))

        coord1_vec = tf.boolean_mask(coord1_vec, tf.logical_not(outside_ind))
        coord2_vec = tf.boolean_mask(coord2_vec, tf.logical_not(outside_ind))

    coord_old_clipped = tf.cast(tf.transpose(tf.pack([coord_old1_clipped, coord_old2_clipped]), [1, 0]), tf.int32)

    # Coordinates of the new image
    coord_new = tf.transpose(tf.cast(tf.pack([coord1_vec, coord2_vec]), tf.int32), [1, 0])

    image_channel_list = tf.split(2, s[2], image)

    image_rotated_channel_list = list()
    for image_channel in image_channel_list:
        image_chan_new_values = tf.gather_nd(tf.squeeze(image_channel), coord_old_clipped)

        if (mode == 'black') or (mode == 'repeat'):
            background_color = 0
        elif mode == 'ones':
            background_color = 1
        elif mode == 'white':
            background_color = 255

        image_rotated_channel_list.append(tf.sparse_to_dense(coord_new, [s[0], s[1]], image_chan_new_values,
                                                             background_color, validate_indices=False))

    image_rotated = tf.transpose(tf.pack(image_rotated_channel_list), [1, 2, 0])

    return image_rotated

def get_save_path(config):
    path = 'weights/model'
    for a, v in config.__dict__.iteritems():
        path += '_' + a + '=' + str(v)
    return path
