import logging
import yaml
import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from contextlib import contextmanager
from json import dumps as pprint

from superpoint.datasets import get_dataset
from superpoint.models import get_model
from superpoint.settings import EXPER_PATH

# Dirty fix until update to 1.6
if tf.__version__ == '1.4.0':
    tf.logging._logger.removeHandler(tf.logging._handler)


def train(config, n_iter, output_dir, checkpoint_name='model.ckpt'):
    with _init_graph(config) as net:
        try:
            net.train(n_iter, output_dir=output_dir)
        except KeyboardInterrupt:
            logging.info('Got Keyboard Interrupt, saving model and closing.')
        net.save(os.path.join(output_dir, checkpoint_name))


def evaluate(config, output_dir, n_iter=None):
    with _init_graph(config) as net:
        net.load(output_dir, last=True)
        results = net.evaluate(config.get('eval_set', 'test'), max_iterations=n_iter)
    return results


def predict(config, output_dir, n_iter):
    pred = []
    data = []
    with _init_graph(config, with_dataset=True) as (net, dataset):
        net.load(output_dir, last=True)
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
    dataset = get_dataset(config['data']['name'])(**config['data'])
    model = get_model(config['model']['name'])(
            data=dataset.get_tf_datasets(), **config['model'])
    model.__enter__()
    if with_dataset:
        yield model, dataset
    else:
        yield model
    model.__exit__()
    tf.reset_default_graph()


def _set_logging(output_dir=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
            '[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(formatter)
    logger.addHandler(h)
    if output_dir:
        h = logging.FileHandler(os.path.join(output_dir, 'log'))
        h.setFormatter(formatter)
        logger.addHandler(h)


def _cli_train(config, args):
    assert 'train_iter' in config
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    if os.path.exists(output_dir):
        raise ValueError('An experiment named {} already exists'.format(args.exper_name))
    os.mkdir(output_dir)
    _set_logging(output_dir)
    logging.info('TRAINING')

    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    train(config, config['train_iter'], output_dir)
    if args.eval:
        _cli_eval(config, args)
    return output_dir


def _cli_eval(config, args):
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    if not os.path.exists(output_dir):
        raise ValueError('Cannot find directory for experiment named {}'.format(
            args.exper_name))
    _set_logging(output_dir)
    logging.info('EVALUATION')

    results = evaluate(config, output_dir, n_iter=config.get('eval_iter'))

    # Print and export results
    logging.info('Evaluation results: \n{}'.format(
        pprint(results, indent=2, default=str)))
    with open(os.path.join(output_dir, 'eval.txt'), 'a') as f:
        f.write('Evaluation for {} dataset:\n'.format(config['data']['name']))
        for r, v in results.items():
            f.write('\t{}:\n\t\t{}\n'.format(r, v))
        f.write('\n')
    return output_dir


# TODO
def _cli_pred(config, args):
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

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
    args.func(config, args)

else:
    _set_logging()
