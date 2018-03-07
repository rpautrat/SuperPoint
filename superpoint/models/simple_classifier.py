import tensorflow as tf
from tensorflow import layers as tfl

from .base_model import BaseModel, Mode


class SimpleClassifier(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {}

    def _model(self, inputs, mode, **config):
        x = inputs['image']

        x = tfl.conv2d(x, 32, 5, activation=tf.nn.relu, padding='SAME', name='conv1')
        x = tfl.max_pooling2d(x, 2, 2, padding='SAME', name='pool1')

        x = tfl.conv2d(x, 64, 5, activation=tf.nn.relu, padding='SAME', name='conv2')
        x = tfl.max_pooling2d(x, 2, 2, padding='SAME', name='pool2')

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
            loss += tf.reduce_sum(
                    tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss

    def _metrics(self, outputs, inputs, **config):
        metrics = {}
        with tf.name_scope('metrics'):
            correct_count = tf.equal(outputs['pred'], inputs['label'])
            correct_count = tf.cast(correct_count, tf.float32)
            metrics['accuracy'] = tf.reduce_mean(correct_count)
        return metrics
