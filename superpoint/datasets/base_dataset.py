import tensorflow as tf

# TODO: force all ops on CPU


class BaseDataset:
    set_names = ['training', 'validation', 'test']

    def __init__(self, **config):
        # Update config
        self.config = getattr(self, 'default_config', {})
        self.config.update(config)

        self.dataset = self._init_dataset(**self.config)

        self.tf_datasets = {}
        self.tf_next = {}
        for t in self.set_names:
            self.tf_datasets[t] = self._get_data(self.dataset, t, **self.config)
            self.tf_next[t] = self.tf_datasets[t].make_one_shot_iterator().get_next()

        # TODO: do we really need to control the GPU alocation for this session ?
        # self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.sess = tf.Session()

    def get_tf_datasets(self):
        return self.tf_datasets

    def get_training_set(self):
        return self._get_set_generator('training')

    def get_validation_set(self):
        return self._get_set_generator('validation')

    def get_test_set(self):
        return self._get_set_generator('test')

    def _get_set_generator(self, set_name):
        while True:
            yield self.sess.run(self.tf_next[set_name])
