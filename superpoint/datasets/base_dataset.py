from abc import ABCMeta, abstractmethod
import tensorflow as tf

from superpoint.utils.tools import dict_update


class BaseDataset(metaclass=ABCMeta):
    """Base model class.

    Arguments:
        config: A dictionary containing the configuration parameters.

    Datasets should inherit from this class and implement the following methods:
        `_init_dataset` and `_get_data`.
    Additionally, the following static attributes should be defined:
        default_config: A dictionary of potential default configuration values (e.g. the
            size of the validation set).
    """
    split_names = ['training', 'validation', 'test']

    @abstractmethod
    def _init_dataset(self, **config):
        """Prepare the dataset for reading.

        This method should configure the dataset for later fetching through `_get_data`,
        such as downloading the data if it is not stored locally, or reading the list of
        data files from disk. Ideally, especially in the case of large images, this
        method shoudl NOT read all the dataset into memory, but rather prepare for faster
        seubsequent fetching.

        Arguments:
            config: A configuration dictionary, given during the object instantiantion.

        Returns:
            An object subsequently passed to `_get_data`, e.g. a list of file paths and
            set splits.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_data(self, dataset, split_name, **config):
        """Reads the dataset splits using the Tensorflow `tf.data` API.

        This method should create a `tf.data.Dataset` object for the given data split,
        with named components defined through a dictionary mapping strings to tensors.

        It typically performs operations such as reading data from a file or from a
        Python generator, shuffling the elements or applying data augmentation to the
        training split. It should however NOT batch the dataset (left to the model).

        Arguments:
            dataset: An object returned by the `_init_dataset` method.
            split_name: A string, the name of the requested split, either `"training"`,
                `"validation"` or `"test"`.
            config: A configuration dictionary, given during the object instantiantion.

        Returns:
            An object of type `tf.data.Dataset` corresponding to the corresponding split.
        """
        raise NotImplementedError

    def get_tf_datasets(self):
        """"Exposes data splits consistent with the Tensorflow `tf.data` API.

        Returns:
            A dictionary mapping split names (`str`, either `"training"`, `"validation"`,
            or `"test"`) to `tf.data.Dataset` objects.
        """
        return self.tf_splits

    def get_training_set(self):
        """Processed training set.

        Returns:
            A generator of elements from the training set as dictionaries mapping
            component names to the corresponding data (e.g. Numpy array).
        """
        return self._get_set_generator('training')

    def get_validation_set(self):
        """Processed validation set.

        Returns:
            A generator of elements from the training set as dictionaries mapping
            component names to the corresponding data (e.g. Numpy array).
        """
        return self._get_set_generator('validation')

    def get_test_set(self):
        """Processed test set.

        Returns:
            A generator of elements from the training set as dictionaries mapping
            component names to the corresponding data (e.g. Numpy array).
        """
        return self._get_set_generator('test')

    def __init__(self, **config):
        # Update config
        self.config = dict_update(getattr(self, 'default_config', {}), config)

        self.dataset = self._init_dataset(**self.config)

        self.tf_splits = {}
        self.tf_next = {}
        with tf.device('/cpu:0'):
            for n in self.split_names:
                self.tf_splits[n] = self._get_data(self.dataset, n, **self.config)
                self.tf_next[n] = self.tf_splits[n].make_one_shot_iterator().get_next()
        self.end_set = tf.errors.OutOfRangeError
        self.sess = tf.Session()

    def _get_set_generator(self, set_name):
        while True:
            yield self.sess.run(self.tf_next[set_name])
