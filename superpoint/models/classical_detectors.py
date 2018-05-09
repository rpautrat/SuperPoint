import tensorflow as tf
import numpy as np
import cv2

from .base_model import BaseModel
from .utils import box_nms


def classical_detector(im, **config):
    if config['method'] == 'harris':
        detections = cv2.cornerHarris(im, 4, 3, 0.04)

    elif config['method'] == 'shi':
        detections = np.zeros(im.shape[:2], np.float)
        thresh = np.linspace(0.0001, 1, 100, endpoint=False)
        for t in thresh:
            corners = cv2.goodFeaturesToTrack(im, 100, t, 5)
            if corners is not None:
                corners = corners.astype(np.int)
                detections[(corners[:, 0, 1], corners[:, 0, 0])] = t

    elif config['method'] == 'fast':
        detector = cv2.FastFeatureDetector_create(10)
        corners = detector.detect(im.astype(np.uint8))
        detections = np.zeros(im.shape[:2], np.float)
        for c in corners:
            detections[tuple(np.flip(np.int0(c.pt), 0))] = c.response

    elif config['method'] == 'random':
        detections = np.random.rand(im.shape[0], im.shape[1])

    return detections.astype(np.float32)


class ClassicalDetectors(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    default_config = {
            'method': 'harris',  # 'shi', 'fast', 'random'
            'threshold': 0.5,
            'nms': 4,
            'top_k': 300,
    }
    trainable = False

    def _model(self, inputs, mode, **config):
        im = inputs['image']
        with tf.device('/cpu:0'):
            prob = tf.map_fn(lambda i: tf.py_func(
                lambda x: classical_detector(x, **config), [i], tf.float32), im)
            prob_nms = prob
            if config['nms']:
                prob_nms = tf.map_fn(lambda p: box_nms(p, config['nms'], min_prob=0.,
                                                       keep_top_k=config['top_k']), prob)
        pred = tf.cast(tf.greater_equal(prob_nms, config['threshold']), tf.int32)
        return {'prob': prob, 'prob_nms': prob_nms, 'pred': pred}

    def _loss(self, outputs, inputs, **config):
        raise NotImplementedError

    def _metrics(self, outputs, inputs, **config):
        pred = outputs['pred']
        labels = inputs['keypoint_map']
        precision = tf.reduce_sum(pred*labels) / tf.reduce_sum(pred)
        recall = tf.reduce_sum(pred*labels) / tf.reduce_sum(labels)
        return {'precision': precision, 'recall': recall}
