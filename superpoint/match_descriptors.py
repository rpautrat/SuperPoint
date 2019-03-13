import numpy as np
import os
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import time
import tensorflow as tf

import experiment
from superpoint.settings import EXPER_PATH
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
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--export_name', type=str, default=None)
    args = parser.parse_args()

    experiment_name = args.experiment_name
    export_name = args.export_name if args.export_name else experiment_name
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    assert 'eval_iter' in config

    output_dir = Path(EXPER_PATH, 'outputs/{}/'.format(export_name))
    if not output_dir.exists():
        os.makedirs(output_dir)
    checkpoint = Path(EXPER_PATH, experiment_name)
    if 'checkpoint' in config:
        checkpoint = Path(checkpoint, config['checkpoint'])

    with experiment._init_graph(config, with_dataset=True) as (net, dataset):
        if net.trainable:
            net.load(str(checkpoint))
        test_set = dataset.get_test_set()

        pbar = tqdm(total=config['eval_iter'] if config['eval_iter'] > 0 else None)
        i = 0
        while True:
            # TODO change homography functions to just take in outputs from
            # network. Get rid of dataset
            try:
                data = next(test_set)
            except dataset.end_set:
                break
            img_file1 = "/home/mmmfarrell/turtle_datasets/snowy_arch1.png"
            img1 = cv2.imread(img_file1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img1 = np.expand_dims(img1, 2)
            img1 = img1.astype(np.float32)

            img_file2 = "/home/mmmfarrell/turtle_datasets/clear_arch1.png"
            img2 = cv2.imread(img_file2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img2 = np.expand_dims(img2, 2)
            img2 = img2.astype(np.float32)

            data1 = {'image': img1}
            data2 = {'image': img2}
            # # im = tf.image.grayscale_to_rgb(data['image'])
            print(img1.shape)
            input()

            # print(data['image'])
            # print(data['image'].shape)
            # print(data['image'].max())
            # print(data['image'].min())

            # cv2.imshow("raw in", data['image'] / 255.)
            cv2.imshow("img1", img1)
            cv2.waitKey(0)

            start_time = time.time()
            pred1 = net.predict(data1, keys=['prob_nms', 'descriptors'])
            end_time = time.time()
            print("Run time:", end_time - start_time)
            # print(pred1['prob_nms'].shape)
            # print(pred1['prob_nms'].max())
            # print(pred1['prob_nms'].min())
            # input()

            start_time = time.time()
            pred2 = net.predict(data2, keys=['prob_nms', 'descriptors'])
            end_time = time.time()
            print("Run time:", end_time - start_time)

            pred = {'prob': pred1['prob_nms'],
                    'warped_prob': pred2['prob_nms'],
                    'desc': pred1['descriptors'],
                    'warped_desc': pred2['descriptors'],
                    'homography': data['homography']}
            print(data['homography'])
            print(data['homography'].shape)

            if not ('name' in data):
                pred.update(data)
            # filename = data['name'].decode('utf-8') if 'name' in data else str(i)
            # filepath = Path(output_dir, '{}.npz'.format(filename))
            # np.savez_compressed(filepath, **pred)
            # i += 1
            # pbar.update(1)
            # if i == config['eval_iter']:
                # break

            output = ev.compute_homography(pred, 1000, 3, False)
            output['image1'] = img1
            output['image2'] = img2

            img = draw_matches(output) / 255.
            plot_imgs([img], titles=["matches"], dpi=200)
            plt.show()

            # print("keys")
            # for key in output.keys():
                # print(key)
            # print(output)
            input()


