import tensorflow as tf
from tensorflow import layers as tfl

from .base_model import BaseModel, Mode


class SimpleClassifier(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {'data_format': 'channels_first'}

    def _model(self, inputs, mode, **config):
        x = inputs['image']
        if config['data_format'] == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])

        params = {'padding': 'SAME', 'data_format': config['data_format']}

        x = tfl.conv2d(x, 32, 5, activation=tf.nn.relu, name='conv1', **params)
        x = tfl.max_pooling2d(x, 2, 2, name='pool1', **params)

        x = tfl.conv2d(x, 64, 5, activation=tf.nn.relu, name='conv2', **params)
        x = tfl.max_pooling2d(x, 2, 2, name='pool2', **params)

        x = tfl.flatten(x)
        x = tfl.dense(x, 1024, activation=tf.nn.relu, name='fc1')
        x = tfl.dense(x, 10, name='fc2')

        if mode == Mode.TRAIN:
            return {'logits': x}
        else:
            return {'logits': x, 'prob': tf.nn.softmax(x), 'pred': tf.argmax(x, axis=-1)}

    def _loss(self, outputs, inputs, **config):
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                    labels=inputs['label'], logits=outputs['logits']))
        return loss

    def _metrics(self, outputs, inputs, **config):
        metrics = {}
        with tf.name_scope('metrics'):
            correct_count = tf.equal(outputs['pred'], inputs['label'])
            correct_count = tf.cast(correct_count, tf.float32)
            metrics['accuracy'] = tf.reduce_mean(correct_count)
        return metrics
