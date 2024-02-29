import tensorflow as tf

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from . import utils


class SuperPoint(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {
            'data_format': 'channels_first',
            'grid_size': 8,
            'detection_threshold': 0.4,
            'descriptor_size': 256,
            'batch_size': 32,
            'learning_rate': 0.001,
            'lambda_d': 250,
            'descriptor_size': 256,
            'positive_margin': 1,
            'negative_margin': 0.2,
            'lambda_loss': 0.0001,
            'nms': 0,
            'top_k': 0,
    }

    def _model(self, inputs, mode, **config):
        config['training'] = (mode == Mode.TRAIN)

        def net(image):
            if config['data_format'] == 'channels_first':
                image = tf.transpose(image, [0, 3, 1, 2])
            features = vgg_backbone(image, **config)
            detections = utils.detector_head(features, **config)
            descriptors = utils.descriptor_head(features, **config)
            return {**detections, **descriptors}

        results = net(inputs['image'])

        if config['training']:
            warped_results = net(inputs['warped']['image'])
            results = {**results, 'warped_results': warped_results,
                       'homography': inputs['warped']['homography']}

        # Apply NMS and get the final prediction
        prob = results['prob']
        if config['nms']:
            prob = tf.map_fn(lambda p: utils.box_nms(
                p, config['nms'], keep_top_k=config['top_k'],
                min_prob=config['detection_threshold']), prob)
            results['prob_nms'] = prob
        results['pred'] = tf.to_int32(tf.greater_equal(
            prob, config['detection_threshold']))

        return results

    def _loss(self, outputs, inputs, **config):
        logits = outputs['logits']
        warped_logits = outputs['warped_results']['logits']
        descriptors = outputs['descriptors_raw']
        warped_descriptors = outputs['warped_results']['descriptors_raw']

        # Switch to 'channels last' once and for all
        if config['data_format'] == 'channels_first':
            logits = tf.transpose(logits, [0, 2, 3, 1])
            warped_logits = tf.transpose(warped_logits, [0, 2, 3, 1])
            descriptors = tf.transpose(descriptors, [0, 2, 3, 1])
            warped_descriptors = tf.transpose(warped_descriptors, [0, 2, 3, 1])

        # Compute the loss for the detector head
        detector_loss = utils.detector_loss(
                inputs['keypoint_map'], logits,
                valid_mask=inputs['valid_mask'], **config)
        warped_detector_loss = utils.detector_loss(
                inputs['warped']['keypoint_map'], warped_logits,
                valid_mask=inputs['warped']['valid_mask'], **config)

        # Compute the loss for the descriptor head
        descriptor_loss = utils.descriptor_loss(
                descriptors, warped_descriptors, outputs['homography'],
                valid_mask=inputs['warped']['valid_mask'], **config)

        tf.summary.scalar('detector_loss1', detector_loss)
        tf.summary.scalar('detector_loss2', warped_detector_loss)
        tf.summary.scalar('detector_loss_full', detector_loss + warped_detector_loss)
        tf.summary.scalar('descriptor_loss', config['lambda_loss'] * descriptor_loss)

        loss = (detector_loss + warped_detector_loss
                + config['lambda_loss'] * descriptor_loss)
        return loss

    def _metrics(self, outputs, inputs, **config):
        pred = inputs['valid_mask'] * outputs['pred']
        labels = inputs['keypoint_map']

        precision = tf.reduce_sum(pred * labels) / tf.reduce_sum(pred)
        recall = tf.reduce_sum(pred * labels) / tf.reduce_sum(labels)

        return {'precision': precision, 'recall': recall}
