import tensorflow as tf
import numpy as np
import cv2

from .base_model import BaseModel


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

    return detections.astype(np.float32)


class ClassicalDetectors(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    default_config = {
            'method': 'harris',  # 'shi', 'fast'
            'threshold': 0.5,
    }
    trainable = False

    def _model(self, inputs, mode, **config):
        im = inputs['image']
        with tf.device('/cpu:0'):
            prob = tf.map_fn(lambda i: tf.py_func(
                lambda x: classical_detector(x, **config), [i], tf.float32), im)
        pred = tf.cast(tf.greater_equal(prob, config['threshold']), tf.int32)
        return {'prob': prob, 'pred': pred}

    def _loss(self, outputs, inputs, **config):
        raise NotImplementedError

    def _metrics(self, outputs, inputs, **config):
        pred = outputs['pred']
        labels = inputs['keypoint_map']
        precision = tf.reduce_sum(pred*labels) / tf.reduce_sum(pred)
        recall = tf.reduce_sum(pred*labels) / tf.reduce_sum(labels)
        return {'precision': precision, 'recall': recall}
