import numpy as np
import tensorflow as tf

from .base_dataset import BaseDataset
from superpoint.datasets import synthetic_dataset


class SyntheticShapes(BaseDataset):
    default_config = {
            'image_size': [240, 320],
            'primitive': 'all',
            'n_background_blobs': 30,
            'validation_size': 100,
            'test_size': 500,
    }
    primitives = [
            'draw_lines',
            'draw_polygon',
            'draw_multiple_polygons',
            'draw_ellipses',
            'draw_star',
            'draw_checkerboard',
            'draw_stripes',
            'draw_cube',
            'gaussian_noise'
    ]

    def _init_dataset(self, **config):
        assert config['primitive'] in ['all']+self.primitives
        return synthetic_dataset

    def _get_data(self, dataset, split_name, **config):
        def _draw_shape(_):
            if config['primitive'] == 'all':
                primitive = np.random.choice(self.primitives)
            else:
                primitive = config['primitive']
            im = dataset.generate_background(config['image_size'],
                                             config['n_background_blobs'])
            points = np.array(getattr(dataset, primitive)(im))
            return im.astype(np.float32), points.astype(np.int32)

        def _preprocess(e_in):
            e_out = {}
            keypoints = tf.reverse(e_in['keypoints'], axis=[-1])
            e_out['keypoint_map'] = tf.scatter_nd(
                    keypoints,
                    tf.ones([tf.shape(keypoints)[0]], dtype=tf.int32),
                    tf.shape(e_in['image']))
            e_out['image'] = tf.expand_dims(e_in['image'], axis=-1)
            return e_out

        def _dummy():
            while True:
                yield 0

        def _set_shapes(im, keypoints):
            im.set_shape(tf.TensorShape(config['image_size']))
            keypoints.set_shape(tf.TensorShape([None, 2]))
            return im, keypoints

        data = tf.data.Dataset.from_generator(_dummy, tf.int32, tf.TensorShape([]))
        data = data.map(
            lambda i: tuple(tf.py_func(_draw_shape, [i], [tf.float32, tf.int32])),
            num_parallel_calls=8)
        data = data.map(_set_shapes)
        data = data.map(lambda im, keypoints: {'image': im, 'keypoints': keypoints})
        data = data.map(_preprocess)

        # Make the length of the validation and test sets finite
        if split_name == 'validation':
            data = data.take(config['validation_size'])
        elif split_name == 'test':
            data = data.take(config['test_size'])

        return data
