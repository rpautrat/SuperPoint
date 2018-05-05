import numpy as np
from os import path as osp
from glob import glob

from superpoint.settings import EXPER_PATH


def get_paths(exper_name):
    """
    Return a list of paths to the outputs of the experiment.
    """
    return glob(osp.join(EXPER_PATH, 'outputs/{}/*.npz'.format(exper_name)))


def compute_tp_fp(data, remove_zero=1e-4, distance_thresh=2, simplified=False):
    """
    Compute the true and false positive rates.
    """
    # Read data
    gt = np.where(data['keypoint_map'])
    gt = np.stack([gt[0], gt[1]], axis=-1)
    n_gt = len(gt)
    prob = data['prob']

    # Filter out predictions with near-zero probability
    mask = np.where(prob > remove_zero)
    prob = prob[mask]
    pred = np.array(mask).T

    # When several detections match the same ground truth point, only pick
    # the one with the highest score  (the others are false positive)
    sort_idx = np.argsort(prob)[::-1]
    prob = prob[sort_idx]
    pred = pred[sort_idx]

    diff = np.expand_dims(pred, axis=1) - np.expand_dims(gt, axis=0)
    dist = np.linalg.norm(diff, axis=-1)
    matches = np.less_equal(dist, distance_thresh)

    tp = []
    matched = np.zeros(len(gt))
    for m in matches:
        correct = np.any(m)
        if correct:
            gt_idx = np.argmax(m)
            tp.append(not matched[gt_idx])
            matched[gt_idx] = 1
        else:
            tp.append(False)
    tp = np.array(tp, bool)
    if simplified:
        tp = np.any(matches, axis=1)  # keeps multiple matches for the same gt point
        n_gt = np.sum(np.minimum(np.sum(matches, axis=0), 1))  # buggy
    fp = np.logical_not(tp)
    return tp, fp, prob, n_gt


def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        idx = ~np.isfinite(c)
        c[idx] = np.where(a[idx] == 0, 1, 0)  # -inf inf NaN
    return c


def compute_pr(exper_name, **kwargs):
    """
    Compute precision and recall.
    """
    # Gather TP and FP for all files
    paths = get_paths(exper_name)
    tp, fp, prob, n_gt = [], [], [], 0
    for path in paths:
        t, f, p, n = compute_tp_fp(np.load(path), **kwargs)
        tp.append(t)
        fp.append(f)
        prob.append(p)
        n_gt += n
    tp = np.concatenate(tp)
    fp = np.concatenate(fp)
    prob = np.concatenate(prob)

    # Sort in descending order of confidence
    sort_idx = np.argsort(prob)[::-1]
    tp = tp[sort_idx]
    fp = fp[sort_idx]
    prob = prob[sort_idx]

    # Cumulative
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = div0(tp_cum, n_gt)
    precision = div0(tp_cum, tp_cum + fp_cum)
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return precision, recall, prob


def compute_mAP(precision, recall):
    """
    Compute average precision.
    """
    return np.sum(precision[1:] * (recall[1:] - recall[:-1]))


def compute_loc_error(exper_name, prob_thresh=0.5, distance_thresh=2):
    """
    Compute the localization error.
    """
    def loc_error_per_image(data):
        # Read data
        gt = np.where(data['keypoint_map'])
        gt = np.stack([gt[0], gt[1]], axis=-1)
        prob = data['prob']

        # Filter out predictions
        mask = np.where(prob > prob_thresh)
        pred = np.array(mask).T
        prob = prob[mask]

        if not len(gt) or not len(pred):
            return []

        diff = np.expand_dims(pred, axis=1) - np.expand_dims(gt, axis=0)
        dist = np.linalg.norm(diff, axis=-1)
        dist = np.min(dist, axis=1)
        correct_dist = dist[np.less_equal(dist, distance_thresh)]
        return correct_dist
    paths = get_paths(exper_name)
    error = []
    for path in paths:
        error.append(loc_error_per_image(np.load(path)))
    return np.mean(np.concatenate(error))


def compute_repeatability(exper_name, prob_thresh=0.5, distance_thresh=3):
    """
    Compute the repeatability. The experiment must contain in its output the prediction
    on 2 images, an original image and a warped version of it, plus the homography
    linking the 2 images.
    """
    def warp_keypoints(keypoints, H):
        warped_col0 = np.add(np.sum(np.multiply(keypoints, H[0, :2]), axis=1), H[0, 2])
        warped_col1 = np.add(np.sum(np.multiply(keypoints, H[1, :2]), axis=1), H[1, 2])
        warped_col2 = np.add(np.sum(np.multiply(keypoints, H[2, :2]), axis=1), H[2, 2])
        warped_col0 = np.divide(warped_col0, warped_col2)
        warped_col1 = np.divide(warped_col1, warped_col2)
        new_keypoints = np.concatenate([warped_col0[:, None], warped_col1[:, None]],
                                       axis=1)
        return new_keypoints

    def filter_keypoints(points, shape):
        """ Keep only the points whose coordinates are
        inside the dimensions of shape. """
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) &\
               (points[:, 1] >= 0) & (points[:, 1] < shape[1])
        return points[mask, :]

    def keep_true_keypoints(points, H, shape):
        """ Keep only the points whose warped coordinates by H
        are still inside shape. """
        warped_points = warp_keypoints(points[:, [1, 0]], H)
        warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
        return points[mask, :]

    paths = get_paths(exper_name)
    repeatability = []
    for path in paths:
        data = np.load(path)
        shape = data['warped_prob'].shape
        H = data['homography']

        # Filter out predictions
        keypoints = np.where(data['prob'] > prob_thresh)
        keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
        warped_keypoints = np.where(data['warped_prob'] > prob_thresh)
        warped_keypoints = np.stack([warped_keypoints[0], warped_keypoints[1]], axis=-1)
        warped_keypoints = keep_true_keypoints(warped_keypoints, np.linalg.inv(H),
                                               data['prob'].shape)

        # Warp the original keypoints with the true homography
        true_warped_keypoints = warp_keypoints(keypoints[:, [1, 0]], H)
        true_warped_keypoints[:, [0, 1]] = true_warped_keypoints[:, [1, 0]]
        true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)

        # Compute the repeatability
        N1 = true_warped_keypoints.shape[0]
        N2 = warped_keypoints.shape[0]
        tile1 = np.tile(np.expand_dims(true_warped_keypoints, 1), (1, N2, 1))
        tile2 = np.tile(warped_keypoints, (N1, 1, 1))
        # both shapes are now N1 x N2 x 2
        norm = np.linalg.norm(tile1 - tile2, ord=None, axis=2)
        count1 = 0
        count2 = 0
        if N2 != 0:
            min1 = np.min(norm, axis=1)
            count1 = np.sum(min1 <= distance_thresh)
        if N1 != 0:
            min2 = np.min(norm, axis=0)
            count2 = np.sum(min2 <= distance_thresh)
        if N1 + N2 > 0:
            repeatability.append((count1 + count2) / (N1 + N2))
        else:
            repeatability.append(1)

    return np.mean(repeatability)


def select_k_points(prob, k=200, min_thresh=0.1):
    """
    Automatically fix the threshold to compute the true keypoints
    and keep only k points (or the most points above min_thresh).
    """

    flat_prob = np.sort(prob, axis=None)
    thresh = min_thresh
    if k < flat_prob.shape[0]:
        thresh = max(flat_prob[-k], min_thresh)
    return np.where(prob >= thresh)
