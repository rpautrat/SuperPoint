import numpy as np
import tensorflow as tf
from pathlib import Path

from .base_dataset import BaseDataset
from superpoint.settings import DATA_PATH, EXPER_PATH


class Coco(BaseDataset):
    default_config = {
        'labels': None,
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': [240, 320]
        }
    }

    def _init_dataset(self, **config):
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

        def _read_points(filename):
            return np.load(filename.decode('utf-8'))['points'].astype(np.int32)

        def _coordinates_to_kmap(image, coordinates, *others):
            kmap = tf.scatter_nd(
                    coordinates,
                    tf.ones([tf.shape(coordinates)[0]], dtype=tf.int32),
                    tf.shape(image)[:2])
            return (image, kmap) + others

        names = tf.data.Dataset.from_tensor_slices(files['names'])
        images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
        images = images.map(_read_image)
        images = images.map(_preprocess)

        if 'label_paths' in files:
            labels = tf.data.Dataset.from_tensor_slices(files['label_paths'])
            labels = labels.map(lambda path: tf.py_func(_read_points, [path], tf.int32))
            labels = labels.map(lambda points: tf.reshape(points, [-1, 2]))
            data = tf.data.Dataset.zip((images, labels, names))
            data = data.map(_coordinates_to_kmap)
            # data = tf.data.Dataset.zip((data, names)
            data = data.map(lambda image, kmap, name:
                            {'image': image, 'keypoint_map': kmap, 'name': name})
        else:
            data = tf.data.Dataset.zip({'image': images, 'name': names})

        if split_name == 'validation':
            data = data.take(config['validation_size'])
        if config['cache_in_memory']:
            tf.logging.info('Caching data, fist access will take some time.')
            data = data.cache()

        return data
