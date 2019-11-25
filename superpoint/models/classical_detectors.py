import tensorflow as tf
import numpy as np
import cv2
import sys

sys.path.append('/cluster/home/pautratr/3d_project/SuperPointPretrainedNetwork')

from .base_model import BaseModel
from .utils import box_nms
from demo_superpoint import SuperPointNet, SuperPointFrontend


def classical_detector(im, **config):
    if config['method'] == 'harris':
        im = np.uint8(im * 255)
        detections = cv2.cornerHarris(im, 4, 3, 0.04)

    elif config['method'] == 'shi':
        im = np.uint8(im * 255)
        detections = np.zeros(im.shape[:2], np.float)
        thresh = np.linspace(0.0001, 1, 600, endpoint=False)
        for t in thresh:
            corners = cv2.goodFeaturesToTrack(im, 600, t, 5)
            if corners is not None:
                corners = corners.astype(np.int)
                detections[(corners[:, 0, 1], corners[:, 0, 0])] = t

    elif config['method'] == 'fast':
        im = np.uint8(im * 255)
        detector = cv2.FastFeatureDetector_create(10)
        corners = detector.detect(im.astype(np.uint8))
        detections = np.zeros(im.shape[:2], np.float)
        for c in corners:
            detections[tuple(np.flip(np.int0(c.pt), 0))] = c.response

    elif config['method'] == 'random':
        detections = np.random.rand(im.shape[0], im.shape[1])

    elif config['method'] == 'pretrained_magic_point':
        weights_path = '/cluster/home/pautratr/3d_project/SuperPointPretrainedNetwork/superpoint_v1.pth'
        fe = SuperPointFrontend(weights_path=weights_path,
                                nms_dist=config['nms'],
                                conf_thresh=0.015,
                                nn_thresh=0.7,
                                cuda=True)
        points, desc, detections = fe.run(im[:, :, 0])

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
