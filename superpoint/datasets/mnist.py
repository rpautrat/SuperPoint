import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from .base_dataset import BaseDataset
from superpoint.settings import DATA_PATH


class Mnist(BaseDataset):
    default_config = {'validation_size': 500}

    def _init_dataset(self, **config):
        return input_data.read_data_sets(os.path.join(DATA_PATH, 'MNIST'),
                                         reshape=False,
                                         validation_size=config['validation_size'])

    def _get_data(self, dataset, split_name, **config):
        if split_name == 'training':
            data = dataset.train
        elif split_name == 'validation':
            data = dataset.validation
        elif split_name == 'test':
            data = dataset.test

        data = tf.data.Dataset.from_tensor_slices(
                {'image': data.images, 'label': data.labels.astype(int)})
        data = data.shuffle(buffer_size=10000)
        return data
