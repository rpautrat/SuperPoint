import yaml
import argparse
import logging
from pathlib import Path

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
import tensorflow as tf  # noqa: E402

from superpoint.models import get_model  # noqa: E402
from superpoint.settings import EXPER_PATH  # noqa: E402


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('export_name', type=str)
    args = parser.parse_args()

    export_name = args.export_name
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    config['model']['data_format'] = 'channels_last'

    export_root_dir = Path(EXPER_PATH, 'saved_models')
    export_root_dir.mkdir(parents=True, exist_ok=True)
    export_dir = Path(export_root_dir, export_name)
    checkpoint_path = Path(EXPER_PATH, export_name)

    with get_model(config['model']['name'])(
            data_shape={'image': [None, None, None, 1]},
            **config['model']) as net:

        net.load(str(checkpoint_path))

        tf.saved_model.simple_save(
                net.sess,
                str(export_dir),
                inputs=net.pred_in,
                outputs=net.pred_out)
