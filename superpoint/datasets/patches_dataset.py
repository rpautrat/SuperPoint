import numpy as np
import tensorflow as tf
from pathlib import Path

from .base_dataset import BaseDataset
from superpoint.models.utils import sample_homography
from superpoint.settings import DATA_PATH


class PatchesDataset(BaseDataset):
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
        base_path = Path(DATA_PATH, 'COCO/patches/')
        folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
        image_paths = []
        warped_image_paths = []
        homographies = []
        for path in folder_paths:
            object_paths = list(path.iterdir())
            num_images = (len(object_paths) - 1) // 2
            for i in range(2, 2 + num_images):
                image_paths.append(str(Path(path, "1.jpg")))
                warped_image_paths.append(str(Path(path, str(i) + ".jpg")))
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

        def _warp_image(image):
            H = sample_homography(tf.shape(image)[:2])
            warped_im = tf.contrib.image.transform(image, H, interpolation="BILINEAR")
            return {'warped_im': warped_im, 'H': H}

        images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
        images = images.map(_read_image)
        images = images.map(_preprocess)
        warped_images = tf.data.Dataset.from_tensor_slices(files['warped_image_paths'])
        warped_images = warped_images.map(_read_image)
        warped_images = warped_images.map(_preprocess)
        homographies = tf.data.Dataset.from_tensor_slices(np.array(files['homography']))

        data = tf.data.Dataset.zip({'image': images, 'warped_image': warped_images,
                                    'homography': homographies})

        if split_name == 'validation':
            data = data.take(config['validation_size'])
        if config['cache_in_memory']:
            tf.logging.info('Caching data, first access will take some time.')
            data = data.cache()

        return data
