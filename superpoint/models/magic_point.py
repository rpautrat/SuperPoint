import tensorflow as tf
from tensorflow import layers as tfl

from .base_model import BaseModel, Mode


def vgg_block(inputs, filters, kernel_size, name, data_format, training=False,
              batch_normalization=True, **params):
    with tf.variable_scope(name):
        x = tfl.conv2d(inputs, filters, kernel_size, name='conv',
                       data_format=data_format, **params)
        if batch_normalization:
            x = tfl.batch_normalization(
                    x, training=training, name='bn', fused=True,
                    axis=1 if data_format == 'channels_first' else -1)
    return x


def vgg_backbone(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'activation': tf.nn.relu, 'batch_normalization': True,
                   'training': config['training']}
    params_pool = {'padding': 'SAME', 'data_format': config['data_format']}

    with tf.variable_scope('vgg'):
        x = vgg_block(inputs, 64, 3, 'conv1_1', **params_conv)
        x = vgg_block(x, 64, 3, 'conv1_2', **params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool1', **params_pool)

        x = vgg_block(x, 64, 3, 'conv2_1', **params_conv)
        x = vgg_block(x, 64, 3, 'conv2_2', **params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool2', **params_pool)

        x = vgg_block(x, 128, 3, 'conv3_1', **params_conv)
        x = vgg_block(x, 128, 3, 'conv3_2', **params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool3', **params_pool)

        x = vgg_block(x, 128, 3, 'conv4_1', **params_conv)
        x = vgg_block(x, 128, 3, 'conv4_2', **params_conv)

    return x


def detector_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'activation': tf.nn.relu, 'batch_normalization': True,
                   'training': config['training']}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    with tf.variable_scope('detector'):
        x = vgg_block(inputs, 256, 3, 'conv1', **params_conv)
        x = vgg_block(inputs, 1+pow(config['grid_size'], 2), 1, 'conv2', **params_conv)

        prob = tf.nn.softmax(x, dim=cindex)
        # Strip the extra “no interest point” dustbin
        prob = prob[:, :-1, :, :] if cfirst else prob[:, :, :, :-1]
        prob = tf.depth_to_space(
                prob, config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')
        prob = tf.squeeze(prob, axis=cindex)

        # Filter maximum per block
        pred = tf.equal(x, tf.reduce_max(x, axis=cindex, keep_dims=True))
        pred = tf.cast(pred, tf.float32)  # as of 1.4: GPU strided_slice only for float32
        pred = pred[:, :-1, :, :] if cfirst else pred[:, :, :, :-1]
        pred = tf.depth_to_space(
                pred, config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')
        pred = tf.cast(tf.squeeze(pred, axis=cindex), tf.int32)

    return {'logits': x, 'prob': prob, 'pred': pred}


class MagicPoint(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {
            'data_format': 'channels_first',
            'grid_size': 8
    }

    # TODO: add homography adaptation for pred, and evaluation ?
    def _model(self, inputs, mode, **config):
        config['training'] = (mode == Mode.TRAIN)

        im = inputs['image']
        if config['data_format'] == 'channels_first':
            im = tf.transpose(im, [0, 3, 1, 2])

        features = vgg_backbone(im, **config)
        outputs = detector_head(features, **config)

        return outputs

    def _loss(self, outputs, inputs, **config):
        cfirst = config['data_format'] == 'channels_first'
        cindex = 1 if cfirst else 3

        # Convert the boolean labels to indices including the "no interest point" dustbin
        labels = tf.cast(tf.expand_dims(inputs['keypoint_map'], axis=cindex), tf.float32)
        labels = tf.space_to_depth(labels, config['grid_size'],
                                   data_format='NCHW' if cfirst else 'NHWC')
        shape = tf.shape(labels)
        shape = tf.concat([shape[:cindex], [1], shape[cindex+1:]], axis=0)
        labels = tf.concat([2*labels, tf.ones(shape)], cindex)  # hacky
        labels = tf.argmax(labels, axis=cindex)

        # Apply the cross entropy
        # `spase_softmax_cross_entropy` seems to only supports channels last
        logits = outputs['logits']
        if cfirst:
            logits = tf.transpose(logits, [0, 2, 3, 1])
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                labels=labels, logits=logits))

        return loss

    # TODO: add involved corner detector metrics
    def _metrics(self, outputs, inputs, **config):
        pred = outputs['pred']
        labels = inputs['keypoint_map']

        precision = tf.reduce_sum(pred*labels) / tf.reduce_sum(pred)
        recall = tf.reduce_sum(pred*labels) / tf.reduce_sum(labels)

        return {'precision': precision, 'recall': recall}
