import numpy as np
import os
import argparse
import yaml
from os import path as osp

import experiment
from superpoint.settings import EXPER_PATH

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('export_name', type=str)
    args = parser.parse_args()

    export_name = args.export_name
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    output_dir = osp.join(EXPER_PATH, 'outputs/{}/'.format(export_name))
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    checkpoint_dir = osp.join(EXPER_PATH, export_name)

    predictions, data = experiment.predict(config, checkpoint_dir, config['eval_iter'])

    for i, (p, d) in enumerate(zip(predictions, data)):
        filepath = osp.join(output_dir, '{}.npz'.format(i))
        outputs = p.copy()
        outputs.update(d)
        np.savez_compressed(filepath, **outputs)
