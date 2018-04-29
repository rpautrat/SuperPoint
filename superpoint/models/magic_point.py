import tensorflow as tf

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from .utils import detector_head, homography_adaptation_batch, box_nms


class MagicPoint(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {
            'data_format': 'channels_first',
            'grid_size': 8,
            'detection_threshold': 0.4,
            'homography_adaptation': {'num': 0},
            'nms': 0,
            'top_k': 0
    }

    def _model(self, inputs, mode, **config):
        config['training'] = (mode == Mode.TRAIN)
        image = inputs['image']

        def net(image):
            if config['data_format'] == 'channels_first':
                image = tf.transpose(image, [0, 3, 1, 2])
            features = vgg_backbone(image, **config)
            outputs = detector_head(features, **config)
            return outputs

        if (mode == Mode.PRED) and config['homography_adaptation']['num']:
            outputs = homography_adaptation_batch(
                    image, net, config['homography_adaptation'])
        else:
            outputs = net(image)

        prob = outputs['prob']
        if config['nms']:
            prob = tf.map_fn(lambda p: box_nms(p, config['nms'],
                                               keep_top_k=config['top_k']), prob)
            outputs['prob_nms'] = prob
        pred = tf.to_int32(tf.greater_equal(prob, config['detection_threshold']))
        outputs['pred'] = pred

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
