import numpy as np
import tensorflow as tf
import cv2
import os
import tarfile
from pathlib import Path
from tqdm import tqdm
import shutil

from .base_dataset import BaseDataset
from superpoint.datasets import synthetic_dataset
from .utils import pipeline
from .utils.pipeline import parse_primitives
from superpoint.settings import DATA_PATH


class SyntheticShapes(BaseDataset):
    default_config = {
            'primitives': 'all',
            'truncate': {},
            'validation_size': -1,
            'test_size': -1,
            'on-the-fly': False,
            'cache_in_memory': False,
            'suffix': None,
            'add_augmentation_to_test_set': False,
            'num_parallel_calls': 10,
            'generation': {
                'split_sizes': {'training': 10000, 'validation': 200, 'test': 500},
                'image_size': [960, 1280],
                'random_seed': 0,
                'params': {
                    'generate_background': {
                        'min_kernel_size': 150, 'max_kernel_size': 500,
                        'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031},
                    'draw_stripes': {'transform_params': (0.1, 0.1)},
                    'draw_multiple_polygons': {'kernel_boundaries': (50, 100)}
                },
            },
            'preprocessing': {
                'resize': [240, 320],
                'blur_size': 11,
            },
            'augmentation': {
                'photometric': {
                    'enable': False,
                    'primitives': 'all',
                    'params': {},
                    'random_order': True,
                },
                'homographic': {
                    'enable': False,
                    'params': {},
                    'valid_border_margin': 0,
                },
            }
    }
    drawing_primitives = [
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

    def dump_primitive_data(self, primitive, tar_path, config):
        temp_dir = Path(os.environ['TMPDIR'], primitive)

        tf.logging.info('Generating tarfile for primitive {}.'.format(primitive))
        synthetic_dataset.set_random_state(np.random.RandomState(
                config['generation']['random_seed']))
        for split, size in self.config['generation']['split_sizes'].items():
            im_dir, pts_dir = [Path(temp_dir, i, split) for i in ['images', 'points']]
            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(size), desc=split, leave=False):
                image = synthetic_dataset.generate_background(
                        config['generation']['image_size'],
                        **config['generation']['params']['generate_background'])
                points = np.array(getattr(synthetic_dataset, primitive)(
                        image, **config['generation']['params'].get(primitive, {})))
                points = np.flip(points, 1)  # reverse convention with opencv

                b = config['preprocessing']['blur_size']
                image = cv2.GaussianBlur(image, (b, b), 0)
                points = (points * np.array(config['preprocessing']['resize'], np.float)
                          / np.array(config['generation']['image_size'], np.float))
                image = cv2.resize(image, tuple(config['preprocessing']['resize'][::-1]),
                                   interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(str(Path(im_dir, '{}.png'.format(i))), image)
                np.save(Path(pts_dir, '{}.npy'.format(i)), points)

        # Pack into a tar file
        tar = tarfile.open(tar_path, mode='w:gz')
        tar.add(temp_dir, arcname=primitive)
        tar.close()
        shutil.rmtree(temp_dir)
        tf.logging.info('Tarfile dumped to {}.'.format(tar_path))

    def _init_dataset(self, **config):
        # Parse drawing primitives
        primitives = parse_primitives(config['primitives'], self.drawing_primitives)

        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
                fn, num_parallel_calls=config['num_parallel_calls'])

        if config['on-the-fly']:
            return None

        basepath = Path(
                DATA_PATH, 'synthetic_shapes' +
                ('_{}'.format(config['suffix']) if config['suffix'] is not None else ''))
        basepath.mkdir(parents=True, exist_ok=True)

        splits = {s: {'images': [], 'points': []}
                  for s in ['training', 'validation', 'test']}
        for primitive in primitives:
            tar_path = Path(basepath, '{}.tar.gz'.format(primitive))
            if not tar_path.exists():
                self.dump_primitive_data(primitive, tar_path, config)

            # Untar locally
            tf.logging.info('Extracting archive for primitive {}.'.format(primitive))
            tar = tarfile.open(tar_path)
            temp_dir = Path(os.environ['TMPDIR'])
            tar.extractall(path=temp_dir)
            tar.close()

            # Gather filenames in all splits, optionally truncate
            truncate = config['truncate'].get(primitive, 1)
            path = Path(temp_dir, primitive)
            for s in splits:
                e = [str(p) for p in Path(path, 'images', s).iterdir()]
                f = [p.replace('images', 'points') for p in e]
                f = [p.replace('.png', '.npy') for p in f]
                splits[s]['images'].extend(e[:int(truncate*len(e))])
                splits[s]['points'].extend(f[:int(truncate*len(f))])

        # Shuffle
        for s in splits:
            perm = np.random.RandomState(0).permutation(len(splits[s]['images']))
            for obj in ['images', 'points']:
                splits[s][obj] = np.array(splits[s][obj])[perm].tolist()
        return splits

    def _get_data(self, filenames, split_name, **config):

        def _gen_shape():
            primitives = parse_primitives(config['primitives'], self.drawing_primitives)
            while True:
                primitive = np.random.choice(primitives)
                image = synthetic_dataset.generate_background(
                        config['generation']['image_size'],
                        **config['generation']['params']['generate_background'])
                points = np.array(getattr(synthetic_dataset, primitive)(
                        image, **config['generation']['params'].get(primitive, {})))
                yield (np.expand_dims(image, axis=-1).astype(np.float32),
                       np.flip(points.astype(np.float32), 1))

        def _read_image(filename):
            image = tf.read_file(filename)
            image = tf.image.decode_png(image, channels=1)
            return tf.cast(image, tf.float32)

        # Python function
        def _read_points(filename):
            return np.load(filename.decode('utf-8')).astype(np.float32)

        if config['on-the-fly']:
            data = tf.data.Dataset.from_generator(
                    _gen_shape, (tf.float32, tf.float32),
                    (tf.TensorShape(config['generation']['image_size']+[1]),
                     tf.TensorShape([None, 2])))
            data = data.map(lambda i, c: pipeline.downsample(
                    i, c, **config['preprocessing']))
        else:
            # Initialize dataset with file names
            data = tf.data.Dataset.from_tensor_slices(
                    (filenames[split_name]['images'], filenames[split_name]['points']))
            # Read image and point coordinates
            data = data.map(
                    lambda image, points:
                    (_read_image(image), tf.py_func(_read_points, [points], tf.float32)))
            data = data.map(lambda image, points: (image, tf.reshape(points, [-1, 2])))

        if split_name == 'validation':
            data = data.take(config['validation_size'])
        elif split_name == 'test':
            data = data.take(config['test_size'])

        data = data.map(lambda image, kp: {'image': image, 'keypoints': kp})
        data = data.map(pipeline.add_dummy_valid_mask)

        if config['cache_in_memory'] and not config['on-the-fly']:
            tf.logging.info('Caching data, fist access will take some time.')
            data = data.cache()

        # Apply augmentation
        if split_name == 'training' or config['add_augmentation_to_test_set']:
            if config['augmentation']['photometric']['enable']:
                data = data.map_parallel(lambda d: pipeline.photometric_augmentation(
                    d, **config['augmentation']['photometric']))
            if config['augmentation']['homographic']['enable']:
                data = data.map_parallel(lambda d: pipeline.homographic_augmentation(
                    d, **config['augmentation']['homographic']))

        # Convert the point coordinates to a dense keypoint map
        data = data.map_parallel(pipeline.add_keypoint_map)
        data = data.map_parallel(lambda d: {**d, 'image': tf.to_float(d['image']) / 255.})

        return data
