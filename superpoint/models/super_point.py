import tensorflow as tf
import numpy as np

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from .utils import (detector_head, descriptor_head, sample_homography,
                    warp_keypoints_to_list, warp_keypoints_to_map,
                    tf_repeat, box_nms)


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
            'positive_margin': 1,
            'negative_margin': 0.2,
            'lambda_loss': 0.0001,
            'nms': 0,
            'top_k': 0,
            'homographies': {'translation': True,
                             'rotation': True,
                             'scaling': True,
                             'perspective': True,
                             'scaling_amplitude': 0.1,
                             'perspective_amplitude': 0.05}
    }

    def _model(self, inputs, mode, **config):
        config['training'] = (mode == Mode.TRAIN)

        im = inputs['image']
        batch_size = config['batch_size']
        shape = tf.shape(im)[1:3]

        with tf.device('/cpu:0'):
            elems = tf.tile(tf.expand_dims(shape, 0), [batch_size, 1])
            H = tf.map_fn(lambda s: sample_homography(s, **config['homographies']),
                          elems, dtype=tf.float32)
            H = tf.reshape(H, [batch_size, 8])
        warped_im = tf.contrib.image.transform(im, H, interpolation="BILINEAR")
        if config['data_format'] == 'channels_first':
            im = tf.transpose(im, [0, 3, 1, 2])
            warped_im = tf.transpose(warped_im, [0, 3, 1, 2])

        features = vgg_backbone(im, **config)
        keypoints = detector_head(features, **config)
        descriptors = descriptor_head(features, **config)

        warped_features = vgg_backbone(warped_im, **config)
        warped_keypoints = detector_head(warped_features, **config)
        warped_descriptors = descriptor_head(warped_features, **config)

        prob = keypoints['prob']
        if config['nms']:
            prob = tf.map_fn(lambda p: box_nms(p, config['nms'],
                                               keep_top_k=config['top_k']), prob)
            keypoints['prob_nms'] = prob
        pred = tf.to_int32(tf.greater_equal(prob, config['detection_threshold']))
        keypoints['pred'] = pred

        prob = warped_keypoints['prob']
        if config['nms']:
            prob = tf.map_fn(lambda p: box_nms(p, config['nms'],
                                               keep_top_k=config['top_k']),
                             warped_keypoints['prob'])
            warped_keypoints['prob_nms'] = prob
        pred = tf.to_int32(tf.greater_equal(prob, config['detection_threshold']))
        warped_keypoints['pred'] = pred

        outputs = {'keypoints': keypoints,
                   'descriptors': descriptors,
                   'warped_keypoints': warped_keypoints,
                   'warped_descriptors': warped_descriptors,
                   'homography': H}

        return outputs

    def _loss(self, outputs, inputs, **config):
        logits = outputs['keypoints']['logits']
        warped_logits = outputs['warped_keypoints']['logits']
        descriptors = outputs['descriptors']['logits']
        warped_descriptors = outputs['warped_descriptors']['logits']

        # Switch to 'channels last' once and for all
        if config['data_format'] == 'channels_first':
            logits = tf.transpose(logits, [0, 2, 3, 1])
            warped_logits = tf.transpose(warped_logits, [0, 2, 3, 1])
            descriptors = tf.transpose(descriptors, [0, 2, 3, 1])
            warped_descriptors = tf.transpose(warped_descriptors, [0, 2, 3, 1])

        # Compute the loss for the keypoints detector
        with tf.device('/cpu:0'):
            warped_labels = tf.map_fn(warp_keypoints_to_map,
                                      (inputs['keypoint_map'], outputs['homography']),
                                      dtype=tf.float32)
        keypoints_loss = self._detector_loss(inputs['keypoint_map'], logits, **config)
        warped_keypoints_loss = self._detector_loss(warped_labels,
                                                    warped_logits,
                                                    **config)

        # Compute the loss for the descriptors
        descriptors_loss = self._descriptors_loss(descriptors, warped_descriptors,
                                                  outputs['homography'], **config)

        loss = keypoints_loss + warped_keypoints_loss +\
            config['lambda_loss'] * descriptors_loss

        return loss

    def _detector_loss(self, keypoint_map_labels, logits, **config):
        labels = tf.cast(tf.expand_dims(keypoint_map_labels, axis=3), tf.float32)
        labels = tf.space_to_depth(labels, config['grid_size'], data_format='NHWC')
        shape = tf.shape(labels)
        shape = tf.concat([shape[:3], [1]], axis=0)
        labels = tf.concat([2*labels, tf.ones(shape)], 3)  # hacky
        labels = tf.argmax(labels, axis=3)

        # Apply the cross entropy
        with tf.device('/cpu:0'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                                   logits=logits)
        return tf.reduce_mean(cross_entropy)

    def _descriptors_loss(self, descriptors, warped_descriptors, H, **config):
        # Compute the position of the center pixel of every cell in the image
        (batch_size, Hc, Wc) = descriptors.shape.as_list()[:3]
        full_shape = [Hc * config['grid_size'], Wc * config['grid_size']]
        row_coord_cells = np.array(range(Hc)) * config['grid_size']\
            + config['grid_size'] // 2
        col_coord_cells = np.array(range(Wc)) * config['grid_size']\
            + config['grid_size'] // 2
        row_coord_cells = np.repeat(row_coord_cells, Wc).reshape((Hc * Wc, 1))
        col_coord_cells = np.tile(col_coord_cells, Hc).reshape((Hc * Wc, 1))
        coord_cells = tf.Variable(np.concatenate([row_coord_cells, col_coord_cells],
                                                 axis=1),
                                  dtype=tf.int32)  # shape = (Hc x Wc, 2)
        map_cells = tf.scatter_nd(coord_cells,
                                  tf.ones([Hc * Wc], dtype=tf.float32),
                                  full_shape)  # shape = (H, W)
        map_cells = tf.tile(tf.expand_dims(map_cells, 0),
                            [batch_size, 1, 1])  # shape = (n_batches, H, W)

        # Compute the position of the warped center pixels
        warped_coord_cells = tf.map_fn(warp_keypoints_to_list,
                                       (map_cells, H),
                                       dtype=tf.float32)
        # shape = (n_batches x Hc x Wc, 2)

        # Reshape the tensors
        coord_cells = tf.tile(coord_cells,
                              [batch_size, 1])  # shape = (n_batches x Hc x Wc, 2)
        coord_cells = tf.reshape(coord_cells, [batch_size, Hc, Wc, 2])
        warped_coord_cells = tf.reshape(warped_coord_cells, [batch_size, Hc, Wc, 2])
        coord_cells = tf.cast(tf_repeat(coord_cells, [1, Hc, Wc, 1]), tf.float32)
        warped_coord_cells = tf.cast(tf.tile(warped_coord_cells, [1, Hc, Wc, 1]),
                                     tf.float32)
        # shape = (n_batches, Hc x Hc, Wc x Wc, 2)

        # Compute the pairwise distances and filter the distances less than the threshold
        cell_distances = tf.norm(coord_cells - warped_coord_cells, axis=-1)
        cell_distances = tf.reshape(cell_distances, [batch_size, Hc, Hc, Wc, Wc])
        cell_distances = tf.transpose(cell_distances, [0, 1, 3, 2, 4])
        # shape = (n_batches, Hc, Wc, Hc, Wc)
        zeros = tf.zeros(cell_distances.shape)
        ones = tf.ones(cell_distances.shape)
        s = tf.where(cell_distances <= config['grid_size'], ones, zeros)

        # Compute the dot product matrix between descriptors: d^t * d'
        descriptors = tf_repeat(descriptors,
                                [1, Hc, Wc, 1])
        warped_descriptors = tf.tile(warped_descriptors,
                                     [1, Hc, Wc, 1])
        # shape = (n_batches, Hc x Hc, Wc x Wc, descriptor_dim)
        dot_product_desc = tf.reduce_sum(tf.multiply(descriptors,
                                                     warped_descriptors), -1)
        dot_product_desc = tf.reshape(dot_product_desc, [batch_size, Hc, Hc, Wc, Wc])
        dot_product_desc = tf.transpose(dot_product_desc, [0, 1, 3, 2, 4])
        # shape = (n_batches, Hc, Wc, Hc, Wc)

        # Compute the loss
        m_pos = config['positive_margin'] * ones
        m_neg = config['negative_margin'] * ones
        return tf.reduce_mean(config['lambda_d'] * s *
                              tf.maximum(zeros, m_pos - dot_product_desc)
                              + (ones - s) * tf.maximum(zeros, dot_product_desc - m_neg))

    def _metrics(self, outputs, inputs, **config):
        pred = outputs['keypoints']['pred']
        labels = inputs['keypoint_map']

        precision = tf.reduce_sum(pred*labels) / tf.reduce_sum(pred)
        recall = tf.reduce_sum(pred*labels) / tf.reduce_sum(labels)

        return {'precision': precision, 'recall': recall}
