import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

from .base_dataset import BaseDataset
from superpoint.models.utils import sample_homography
from superpoint.settings import DATA_PATH


class PatchesDataset(BaseDataset):
    default_config = {
        'dataset': 'hpatches',  # or 'coco'
        'alteration': 'all',  # 'all', 'i' for illumination or 'v' for viewpoint
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': [240, 320]
        }
    }

    def _init_dataset(self, **config):
        dataset_folder = 'COCO/patches' if config['dataset'] == 'coco' else 'HPatches'
        base_path = Path(DATA_PATH, dataset_folder)
        folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
        image_paths = []
        warped_image_paths = []
        homographies = []
        for path in folder_paths:
            if config['alteration'] == 'i' and path.stem[0] != 'i':
                continue
            if config['alteration'] == 'v' and path.stem[0] != 'v':
                continue
            num_images = 1 if config['dataset'] == 'coco' else 5
            file_ext = '.ppm' if config['dataset'] == 'hpatches' else '.jpg'
            for i in range(2, 2 + num_images):
                image_paths.append(str(Path(path, "1" + file_ext)))
                warped_image_paths.append(str(Path(path, str(i) + file_ext)))
                homographies.append(np.loadtxt(str(Path(path, "H_1_" + str(i)))))
        if config['truncate']:
            image_paths = image_paths[:config['truncate']]
            warped_image_paths = warped_image_paths[:config['truncate']]
            homographies = homographies[:config['truncate']]
        files = {'image_paths': image_paths,
                 'warped_image_paths': warped_image_paths,
                 'homography': homographies}
        return files

    def _get_data(self, files, split_name, **config):
        def _read_image(path):
            return cv2.imread(path.decode('utf-8'))

        def _scale_preserving_resize(image):
            target_size = tf.convert_to_tensor(config['preprocessing']['resize'])
            scales = tf.to_float(tf.divide(target_size, tf.shape(image)[:2]))
            new_size = tf.to_float(tf.shape(image)[:2]) * tf.reduce_max(scales)
            image = tf.image.resize_images(image, tf.to_int32(new_size),
                                           method=tf.image.ResizeMethod.BILINEAR)
            return tf.image.resize_image_with_crop_or_pad(image, target_size[0],
                                                          target_size[1])

        def _preprocess(image):
            tf.Tensor.set_shape(image, [None, None, 3])
            image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing']['resize']:
                image = _scale_preserving_resize(image)
            return tf.to_float(image)

        def _warp_image(image):
            H = sample_homography(tf.shape(image)[:2])
            warped_im = tf.contrib.image.transform(image, H, interpolation="BILINEAR")
            return {'warped_im': warped_im, 'H': H}

        def _adapt_homography_to_preprocessing(zip_data):
            image = zip_data['image']
            H = tf.cast(zip_data['homography'], tf.float32)
            target_size = tf.convert_to_tensor(config['preprocessing']['resize'])
            s = tf.reduce_max(tf.cast(tf.divide(target_size,
                                                tf.shape(image)[:2]), tf.float32))
            down_scale = tf.diag(tf.stack([1/s, 1/s, tf.constant(1.)]))
            up_scale = tf.diag(tf.stack([s, s, tf.constant(1.)]))
            H = tf.matmul(up_scale, tf.matmul(H, down_scale))
            return H

        images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
        images = images.map(lambda path: tf.py_func(_read_image, [path], tf.uint8))
        homographies = tf.data.Dataset.from_tensor_slices(np.array(files['homography']))
        if config['preprocessing']['resize']:
            homographies = tf.data.Dataset.zip({'image': images,
                                                'homography': homographies})
            homographies = homographies.map(_adapt_homography_to_preprocessing)
        images = images.map(_preprocess)
        warped_images = tf.data.Dataset.from_tensor_slices(files['warped_image_paths'])
        warped_images = warped_images.map(lambda path: tf.py_func(_read_image,
                                                                  [path],
                                                                  tf.uint8))
        warped_images = warped_images.map(_preprocess)

        data = tf.data.Dataset.zip({'image': images, 'warped_image': warped_images,
                                    'homography': homographies})

        if split_name == 'validation':
            data = data.take(config['validation_size'])
        if config['cache_in_memory']:
            tf.logging.info('Caching data, first access will take some time.')
            data = data.cache()

        return data
