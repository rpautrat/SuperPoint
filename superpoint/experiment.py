import logging
import yaml
import os
import argparse
import numpy as np
from contextlib import contextmanager
from json import dumps as pprint

from superpoint.datasets import get_dataset
from superpoint.models import get_model
from superpoint.utils.stdout_capturing import capture_outputs
from superpoint.settings import EXPER_PATH

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Bugfix for TF 1.4
import tensorflow as tf  # noqa: E402


# Dirty fix until update to 1.6
if tf.__version__ == '1.4.0':
    tf.logging._logger.removeHandler(tf.logging._handler)

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def train(config, n_iter, output_dir, checkpoint_name='model.ckpt'):
    with _init_graph(config) as net:
        try:
            net.train(n_iter, output_dir=output_dir)
        except KeyboardInterrupt:
            logging.info('Got Keyboard Interrupt, saving model and closing.')
        net.save(os.path.join(output_dir, checkpoint_name))


def evaluate(config, output_dir, n_iter=None):
    with _init_graph(config) as net:
        net.load(output_dir)
        results = net.evaluate(config.get('eval_set', 'test'), max_iterations=n_iter)
    return results


def predict(config, output_dir, n_iter):
    pred = []
    data = []
    with _init_graph(config, with_dataset=True) as (net, dataset):
        net.load(output_dir)
        test_set = dataset.get_test_set()
        for _ in range(n_iter):
            data.append(next(test_set))
            pred.append(net.predict(data[-1], keys='*'))
    return pred, data


def set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


@contextmanager
def _init_graph(config, with_dataset=False):
    set_seed(config.get('seed', int.from_bytes(os.urandom(4), byteorder='big')))
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    logging.info('Number of GPUs detected: {}'.format(n_gpus))

    dataset = get_dataset(config['data']['name'])(**config['data'])
    model = get_model(config['model']['name'])(
            data=dataset.get_tf_datasets(), n_gpus=n_gpus, **config['model'])
    model.__enter__()
    if with_dataset:
        yield model, dataset
    else:
        yield model
    model.__exit__()
    tf.reset_default_graph()


def _cli_train(config, output_dir, args):
    assert 'train_iter' in config

    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    train(config, config['train_iter'], output_dir)

    if args.eval:
        _cli_eval(config, output_dir, args)


def _cli_eval(config, output_dir, args):
    # Load model config from previous experiment
    with open(os.path.join(output_dir, 'config.yml'), 'r') as f:
        model_config = yaml.load(f)['model']
    model_config.update(config.get('model', {}))
    config['model'] = model_config

    results = evaluate(config, output_dir, n_iter=config.get('eval_iter'))

    # Print and export results
    logging.info('Evaluation results: \n{}'.format(
        pprint(results, indent=2, default=str)))
    with open(os.path.join(output_dir, 'eval.txt'), 'a') as f:
        f.write('Evaluation for {} dataset:\n'.format(config['data']['name']))
        for r, v in results.items():
            f.write('\t{}:\n\t\t{}\n'.format(r, v))
        f.write('\n')


# TODO
def _cli_pred(config, args):
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    p_train = subparsers.add_parser('train')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.set_defaults(func=_cli_train)

    # Evaluation command
    p_train = subparsers.add_parser('evaluate')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.set_defaults(func=_cli_eval)

    # Inference command
    p_train = subparsers.add_parser('predict')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.set_defaults(func=_cli_pred)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with capture_outputs(os.path.join(output_dir, 'log')):
        logging.info('Running command {}'.format(args.command.upper()))
        args.func(config, output_dir, args)
