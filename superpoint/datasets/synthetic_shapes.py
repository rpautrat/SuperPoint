import numpy as np
import tensorflow as tf

from .base_dataset import BaseDataset
from .synthetic_dataset import SyntheticDataset


class SyntheticShapes(BaseDataset):
    default_config = {
            'image_size': [240, 320],
            'primitive': 'all',
            'validation_size': 100,
            'test_size': 500,
    }
    primitives = [
            'draw_lines',
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
        return SyntheticDataset()

    def _get_data(self, dataset, split_name, **config):
        def _gen_shape():
            while True:
                if config['primitive'] == 'all':
                    primitive = np.random.choice(self.primitives)
                else:
                    primitive = config['primitive']
                im = dataset.generate_background(config['image_size'])
                points = np.array(getattr(dataset, primitive)(im))
                yield {'image': im, 'keypoints': points}

        def _preprocess(e_in):
            e_out = {}
            keypoints = tf.reverse(e_in['keypoints'], axis=[-1])
            e_out['keypoint_map'] = tf.scatter_nd(
                    keypoints,
                    tf.ones([tf.shape(keypoints)[0]], dtype=tf.int32),
                    tf.shape(e_in['image']))
            e_out['image'] = tf.expand_dims(e_in['image'], axis=-1)
            return e_out

        data = tf.data.Dataset.from_generator(
                _gen_shape,
                {'image': tf.float32, 'keypoints': tf.int32},
                {'image': tf.TensorShape(config['image_size']),
                 'keypoints': tf.TensorShape([None, 2])})
        data = data.map(_preprocess)

        # Make the length of the validation and test sets finite
        if split_name == 'validation':
            data = data.take(config['validation_size'])
        elif split_name == 'test':
            data = data.take(config['test_size'])

        return data
