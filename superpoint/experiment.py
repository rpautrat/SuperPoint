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

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
import tensorflow as tf  # noqa: E402


def train(config, n_iter, output_dir, pretrained_dir=None,
          checkpoint_name='model.ckpt'):
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    with _init_graph(config) as net:
        if pretrained_dir is not None:
            net.load(pretrained_dir)
        try:
            net.train(n_iter, output_dir=output_dir,
                      validation_interval=config.get('validation_interval', 100),
                      save_interval=config.get('save_interval', None),
                      checkpoint_path=checkpoint_path,
                      keep_checkpoints=config.get('keep_checkpoints', 1))
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
        if net.trainable:
            net.load(output_dir)
        test_set = dataset.get_test_set()
        for _ in range(n_iter):
            data.append(next(test_set))
            pred.append(net.predict(data[-1], keys='*'))
    return pred, data


def set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


def get_num_gpus():
    return len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))


@contextmanager
def _init_graph(config, with_dataset=False):
    set_seed(config.get('seed', int.from_bytes(os.urandom(4), byteorder='big')))
    n_gpus = get_num_gpus()
    logging.info('Number of GPUs detected: {}'.format(n_gpus))

    dataset = get_dataset(config['data']['name'])(**config['data'])
    model = get_model(config['model']['name'])(
            data={} if with_dataset else dataset.get_tf_datasets(),
            n_gpus=n_gpus, **config['model'])
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
        
    if args.pretrained_model is not None:
        pretrained_dir = os.path.join(EXPER_PATH, args.pretrained_model)
        if not os.path.exists(pretrained_dir):
            raise ValueError("Missing pretrained model: " + pretrained_dir)
    else:
        pretrained_dir = None
        
    train(config, config['train_iter'], output_dir, pretrained_dir)

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
    p_train.add_argument('--pretrained_model', type=str, default=None)
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
