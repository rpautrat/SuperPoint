import tensorflow as tf
from .backbones.vgg import vgg_block


def detector_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'activation': tf.nn.relu, 'batch_normalization': True,
                   'training': config['training']}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    with tf.variable_scope('detector'):
        x = vgg_block(inputs, 256, 3, 'conv1', **params_conv)
        x = vgg_block(inputs, 1+pow(config['grid_size'], 2), 1, 'conv2', **params_conv)

        prob = tf.nn.softmax(x, axis=cindex)
        # Strip the extra “no interest point” dustbin
        prob = prob[:, :-1, :, :] if cfirst else prob[:, :, :, :-1]
        prob = tf.depth_to_space(
                prob, config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')
        prob = tf.squeeze(prob, axis=cindex)

        pred = tf.to_int32(tf.greater_equal(prob, config['detection_threshold']))

    return {'logits': x, 'prob': prob, 'pred': pred}
