import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

from .base_dataset import BaseDataset
from .utils import pipeline
from superpoint.models.homographies import sample_homography
from superpoint.settings import DATA_PATH


class PatchesDataset(BaseDataset):
    default_config = {
        'dataset': 'hpatches',  # or 'coco'
        'alteration': 'all',  # 'all', 'i' for illumination or 'v' for viewpoint
        'cache_in_memory': False,
        'truncate': None,
        'preprocessing': {
            'resize': False
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

        def _preprocess(image):
            tf.Tensor.set_shape(image, [None, None, 3])
            image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing']['resize']:
                image = pipeline.ratio_preserving_resize(image,
                                                         **config['preprocessing'])
            return tf.to_float(image)

        def _adapt_homography_to_preprocessing(zip_data):
            H = tf.cast(zip_data['homography'], tf.float32)
            source_size = tf.cast(zip_data['shape'], tf.float32)
            source_warped_size = tf.cast(zip_data['warped_shape'], tf.float32)
            target_size = tf.cast(tf.convert_to_tensor(config['preprocessing']['resize']),
                                  tf.float32)

            # Compute the scaling ratio due to the resizing for both images
            s = tf.reduce_max(tf.divide(target_size, source_size))
            up_scale = tf.diag(tf.stack([1. / s, 1. / s, tf.constant(1.)]))
            warped_s = tf.reduce_max(tf.divide(target_size, source_warped_size))
            down_scale = tf.diag(tf.stack([warped_s, warped_s, tf.constant(1.)]))

            # Compute the translation due to the crop for both images
            pad_y = tf.to_int32(((source_size[0] * s - target_size[0]) / tf.constant(2.0)))
            pad_x = tf.to_int32(((source_size[1] * s - target_size[1]) / tf.constant(2.0)))
            translation = tf.stack([tf.constant(1), tf.constant(0), pad_x, 
                                    tf.constant(0), tf.constant(1), pad_y,
                                    tf.constant(0),tf.constant(0), tf.constant(1)])
            translation = tf.to_float(tf.reshape(translation, [3,3]))
            pad_y = tf.to_int32(((source_warped_size[0] * warped_s - target_size[0])
                                 / tf.constant(2.0)))
            pad_x = tf.to_int32(((source_warped_size[1] * warped_s - target_size[1])
                                 / tf.constant(2.0)))
            warped_translation = tf.stack([tf.constant(1), tf.constant(0), -pad_x, 
                                           tf.constant(0), tf.constant(1), -pad_y,
                                           tf.constant(0),tf.constant(0), tf.constant(1)])
            warped_translation = tf.to_float(tf.reshape(warped_translation, [3,3]))

            H = warped_translation @ down_scale @ H @ up_scale @ translation
            return H

        def _get_shape(image):
            return tf.shape(image)[:2]

        images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
        images = images.map(lambda path: tf.py_func(_read_image, [path], tf.uint8))
        homographies = tf.data.Dataset.from_tensor_slices(np.array(files['homography']))
        warped_images = tf.data.Dataset.from_tensor_slices(files['warped_image_paths'])
        warped_images = warped_images.map(lambda path: tf.py_func(_read_image,
                                                                  [path],
                                                                  tf.uint8))       
        if config['preprocessing']['resize']:
            shapes = images.map(_get_shape)
            warped_shapes = warped_images.map(_get_shape)
            homographies = tf.data.Dataset.zip({'homography': homographies,
                                                'shape': shapes,
                                                'warped_shape': warped_shapes})
            homographies = homographies.map(_adapt_homography_to_preprocessing)
            
        images = images.map(_preprocess)
        warped_images = warped_images.map(_preprocess)

        images = images.map(lambda img: tf.to_float(img) / 255.)
        warped_images = warped_images.map(lambda img: tf.to_float(img) / 255.)

        data = tf.data.Dataset.zip({'image': images, 'warped_image': warped_images,
                                    'homography': homographies})

        return data
