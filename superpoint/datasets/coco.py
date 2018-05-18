import numpy as np
import tensorflow as tf
from pathlib import Path

from .base_dataset import BaseDataset
from superpoint.datasets.utils import augmentation as daug
from superpoint.settings import DATA_PATH, EXPER_PATH


class Coco(BaseDataset):
    default_config = {
        'labels': None,
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': [240, 320]
        },
        'augmentation': {
            'enable': False,
            'primitives': 'all',
            'params': {},
        }
    }

    def parse_primitives(self, names, all_primitives):
        p = all_primitives if (names == 'all') \
                else (names if isinstance(names, list) else [names])
        assert set(p) <= set(all_primitives)
        return p

    def _init_dataset(self, **config):
        config['augmentation']['primitives'] = self.parse_primitives(
                config['augmentation']['primitives'], daug.augmentations)

        base_path = Path(DATA_PATH, 'COCO/train2014/')
        image_paths = list(base_path.iterdir())
        if config['truncate']:
            image_paths = image_paths[:config['truncate']]
        names = [p.stem for p in image_paths]
        image_paths = [str(p) for p in image_paths]
        files = {'image_paths': image_paths, 'names': names}

        if config['labels']:
            label_paths = []
            for n in names:
                p = Path(EXPER_PATH, config['labels'], '{}.npz'.format(n))
                assert p.exists(), 'Image {} has no corresponding label {}'.format(n, p)
                label_paths.append(str(p))
            files['label_paths'] = label_paths

        return files

    def _get_data(self, files, split_name, **config):

        def _read_image(path):
            image = tf.read_file(path)
            image = tf.image.decode_png(image, channels=3)
            return image

        def _scale_preserving_resize(image):
            target_size = tf.convert_to_tensor(config['preprocessing']['resize'])
            scales = tf.to_float(tf.divide(target_size, tf.shape(image)[:2]))
            new_size = tf.to_float(tf.shape(image)[:2]) * tf.reduce_max(scales)
            image = tf.image.resize_images(image, tf.to_int32(new_size),
                                           method=tf.image.ResizeMethod.BILINEAR)
            return tf.image.resize_image_with_crop_or_pad(image, target_size[0],
                                                          target_size[1])

        def _preprocess(image):
            image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing']['resize']:
                image = _scale_preserving_resize(image)
            return image

        # Python function
        def _read_points(filename):
            return np.load(filename.decode('utf-8'))['points'].astype(np.int32)

        def _add_keypoint_map(data):
            kp = tf.to_int32(tf.round(data['keypoints']))
            kmap = tf.scatter_nd(
                    kp, tf.ones([tf.shape(kp)[0]], dtype=tf.int32),
                    tf.shape(data['image'])[:2])
            return {**data, **{'keypoint_map': kmap}}

        # Python function
        def _augmentation(image, points):
            image = image[:, :, 0]
            points = np.flip(points, -1)
            for primitive in config['augmentation']['primitives']:
                image, points = getattr(daug, primitive)(
                        image, points,
                        **config['augmentation']['params'].get(primitive, {}))
            return (image[..., np.newaxis].astype(np.float32),
                    np.flip(points, -1).astype(np.float32))

        names = tf.data.Dataset.from_tensor_slices(files['names'])
        images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
        images = images.map(_read_image)
        images = images.map(_preprocess)
        data = tf.data.Dataset.zip({'image': images, 'name': names})

        # Add keypoints
        if 'label_paths' in files:
            kp = tf.data.Dataset.from_tensor_slices(files['label_paths'])
            kp = kp.map(lambda path: tf.py_func(_read_points, [path], tf.int32))
            kp = kp.map(lambda points: tf.reshape(points, [-1, 2]))
            data = tf.data.Dataset.zip((data, kp)).map(
                    lambda d, k: {**d, **{'keypoints': k}})

        # Keep only the first elements for validation
        if split_name == 'validation':
            data = data.take(config['validation_size'])

        # Cache to avoid always reading from disk
        if config['cache_in_memory']:
            tf.logging.info('Caching data, fist access will take some time.')
            data = data.cache()

        # Data augmentation
        if config['augmentation']['enable'] and 'label_paths' in files \
                and split_name == 'training':
            augmented = data.map(
                    lambda d: tuple(tf.py_func(_augmentation,
                                               [d['image'], d['keypoints']],
                                               [tf.float32, tf.float32])))
            augmented = augmented.map(lambda i, k: (i, tf.reshape(k, [-1, 2])))
            data = tf.data.Dataset.zip((data, augmented)).map(
                    lambda d, a: {**d, **dict(zip(('image', 'keypoints'), a))})

        # Generate the keypoint map
        if 'label_paths' in files:
            data = data.map(_add_keypoint_map)

        return data
