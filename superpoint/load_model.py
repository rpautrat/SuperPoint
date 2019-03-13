import numpy as np
import yaml
import argparse
import logging
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import time

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
import tensorflow as tf  # noqa: E402

from superpoint.models import get_model  # noqa: E402
from superpoint.settings import EXPER_PATH  # noqa: E402
import superpoint.evaluations.my_descriptor_evaluation as ev

def plot_imgs(imgs, titles=None, cmap='brg', ylabel='', normalize=False, ax=None, dpi=100):
    n = len(imgs)
    if not isinstance(cmap, list):
        cmap = [cmap]*n
    if ax is None:
        _, ax = plt.subplots(1, n, figsize=(6*n, 6), dpi=dpi)
        if n == 1:
            ax = [ax]
    else:
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) == len(imgs)
    for i in range(n):
        if imgs[i].shape[-1] == 3:
            imgs[i] = imgs[i][..., ::-1]  # BGR to RGB
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmap[i]),
                     vmin=None if normalize else 0,
                     vmax=None if normalize else 1)
        if titles:
            ax[i].set_title(titles[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    ax[0].set_ylabel(ylabel)
    plt.tight_layout()

def draw_matches(data):
    keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints1']]
    keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints2']]
    inliers = data['inliers'].astype(bool)
    matches = np.array(data['matches'])[inliers].tolist()
    img1 = np.concatenate([output['image1'], output['image1'], output['image1']], axis=2)
    img2 = np.concatenate([output['image2'], output['image2'], output['image2']], axis=2)
    return cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                           None, matchColor=(0,255,0), singlePointColor=(0, 0, 255))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('export_name', type=str)
    args = parser.parse_args()

    export_name = args.export_name
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    config['model']['data_format'] = 'channels_last'

    export_name = "sp_v5"
    export_root_dir = Path(EXPER_PATH, 'saved_models')
    export_root_dir.mkdir(parents=True, exist_ok=True)
    export_dir = Path(export_root_dir, export_name)
    checkpoint_path = Path(EXPER_PATH, export_name)

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(sess,
                [tf.saved_model.tag_constants.SERVING], str(export_dir))

        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        print(input_img_tensor)
        output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
        print(output_prob_nms_tensor)
        output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')
        print(output_prob_nms_tensor)

        img_file1 = "/home/mmmfarrell/turtle_datasets/snowy_arch1.png"
        img1 = cv2.imread(img_file1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = np.expand_dims(img1, 2)
        img1 = img1.astype(np.float32)

        start_time = time.time()
        out1 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                feed_dict={input_img_tensor: np.expand_dims(img1, 0)})
        end_time = time.time()
        print("Run time:", end_time - start_time)

        img_file2 = "/home/mmmfarrell/turtle_datasets/clear_arch1.png"
        img2 = cv2.imread(img_file2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = np.expand_dims(img2, 2)
        img2 = img2.astype(np.float32)

        start_time = time.time()
        out2 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                feed_dict={input_img_tensor: np.expand_dims(img2, 0)})
        end_time = time.time()
        print("Run time:", end_time - start_time)

        data = {}
        data['homography'] = np.eye(3)
        pred = {'prob': np.squeeze(out1[0]),
                'warped_prob': np.squeeze(out2[0]),
                'desc': np.squeeze(out1[1]),
                'warped_desc': np.squeeze(out2[1]),
                'homography': data['homography']}

        output = ev.compute_homography(pred, 1000, 3, False)
        output['image1'] = img1
        output['image2'] = img2

        img = draw_matches(output) / 255.
        plot_imgs([img], titles=["matches"], dpi=200)
        plt.show()
    
