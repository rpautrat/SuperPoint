import tensorflow as tf
from math import pi

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from .utils import (detector_head, descriptor_head, sample_homography,
                    warp_keypoints_to_map, warp_keypoints, box_nms)


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
            'homography_sampling': {
                'translation': True,
                'rotation': True,
                'scaling': True,
                'perspective': True,
                'scaling_amplitude': 0.1,
                'perspective_amplitude': 0.05,
                'patch_ratio': 0.75,
                'max_angle': pi}
    }

    def _model(self, inputs, mode, **config):
        config['training'] = (mode == Mode.TRAIN)

        im = inputs['image']
        batch_size = config['batch_size']
        shape = tf.shape(im)[1:3]

        # Sample random homographies and warp the original image
        elems = tf.tile(tf.expand_dims(shape, 0), [batch_size, 1])
        H = tf.map_fn(lambda s: sample_homography(s, **config['homography_sampling']),
                      elems, dtype=tf.float32)
        H = tf.reshape(H, [batch_size, 8])
        warped_im = tf.contrib.image.transform(im, H, interpolation="BILINEAR")

        def net(image):
            if config['data_format'] == 'channels_first':
                image = tf.transpose(image, [0, 3, 1, 2])
            features = vgg_backbone(image, **config)
            detections = detector_head(features, **config)
            descriptors = descriptor_head(features, **config)
            return {**detections, **descriptors}

        results = net(im)
        warped_results = net(warped_im)

        # Apply NMS and get the final prediction
        prob = results['prob']
        if config['nms']:
            prob = tf.map_fn(lambda p: box_nms(p, config['nms'],
                                               keep_top_k=config['top_k']), prob)
            results['prob_nms'] = prob
        results['pred'] = tf.to_int32(tf.greater_equal(
            prob, config['detection_threshold']))

        return {**results, 'warped_results': warped_results, 'homography': H}

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

        # Compute the loss for the keypoints detector
        warped_labels = tf.map_fn(warp_keypoints_to_map,
                                  (inputs['keypoint_map'], outputs['homography']),
                                  dtype=tf.float32)
        keypoints_loss = self._detector_loss(inputs['keypoint_map'], logits, **config)
        warped_keypoints_loss = self._detector_loss(warped_labels,
                                                    warped_logits,
                                                    **config)

        # Compute the loss for the descriptors
        descriptor_loss = self._descriptor_loss(descriptors, warped_descriptors,
                                                outputs['homography'], **config)

        loss = (keypoints_loss + warped_keypoints_loss
                + config['lambda_loss'] * descriptor_loss)
        return loss

    def _detector_loss(self, keypoint_map_labels, logits, **config):
        labels = tf.cast(tf.expand_dims(keypoint_map_labels, axis=3), tf.float32)
        labels = tf.space_to_depth(labels, config['grid_size'], data_format='NHWC')
        shape = tf.concat([tf.shape(labels)[:3], [1]], axis=0)
        labels = tf.concat([2*labels, tf.ones(shape)], 3)  # hacky
        labels = tf.argmax(labels, axis=3)

        # Apply the cross entropy
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                               logits=logits)
        return tf.reduce_mean(cross_entropy)

    def _descriptor_loss(self, descriptors, warped_descriptors, homographies, **config):
        # Compute the position of the center pixel of every cell in the image
        (batch_size, Hc, Wc) = tf.unstack(tf.to_int32(tf.shape(descriptors)[:3]))
        coord_cells = tf.stack(tf.meshgrid(
            tf.range(Hc), tf.range(Wc), indexing='ij'), axis=-1)
        coord_cells = coord_cells + config['grid_size'] // 2  # (Hc, Wc, 2)
        # coord_cells is now a grid containing the coordinates of the Hc x Wc
        # center pixels of the 8x8 cells of the image

        # Compute the position of the warped center pixels
        warped_coord_cells = warp_keypoints(
                tf.reshape(coord_cells, [-1, 2]), homographies)  # (N, Hc x Wc, 2)
        # warped_coord_cells is now a list of the warped coordinates of all the center
        # pixels of the 8x8 cells of the image

        # Compute the pairwise distances and filter the the ones less than a threshold
        # The distance is just the pairwise norm of the difference of the two grids
        # Using shape broadcasting, cell_distances has shape (N, Hc, Wc, Hc, Wc)
        coord_cells = tf.to_float(tf.reshape(coord_cells, [1, Hc, Wc, 1, 1, 2]))
        warped_coord_cells = tf.reshape(warped_coord_cells,
                                        [batch_size, 1, 1, Hc, Wc, 2])
        cell_distances = tf.norm(coord_cells - warped_coord_cells, axis=-1)
        s = tf.to_float(tf.less_equal(cell_distances, config['grid_size']))
        # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by the
        # homography is at a distance from (h', w') less than config['grid_size']
        # and 0 otherwise

        # Compute the pairwise dot product between descriptors: d^t * d'
        descriptors = tf.reshape(descriptors, [batch_size, Hc, Wc, 1, 1, -1])
        warped_descriptors = tf.reshape(warped_descriptors,
                                        [batch_size, 1, 1, Hc, Wc, -1])
        dot_product_desc = tf.reduce_sum(descriptors * warped_descriptors, -1)
        # dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
        # descriptor at position (h, w) in the original descriptors map and the
        # descriptor at position (h', w') in the warped image

        # Compute the loss
        positive_dist = tf.maximum(0., config['positive_margin'] - dot_product_desc)
        negative_dist = tf.maximum(0., dot_product_desc - config['negative_margin'])
        loss = config['lambda_d'] * s * positive_dist + (1 - s) * negative_dist
        return tf.reduce_mean(loss)

    def _metrics(self, outputs, inputs, **config):
        pred = outputs['pred']
        labels = inputs['keypoint_map']

        precision = tf.reduce_sum(pred*labels) / tf.reduce_sum(pred)
        recall = tf.reduce_sum(pred*labels) / tf.reduce_sum(labels)

        return {'precision': precision, 'recall': recall}
