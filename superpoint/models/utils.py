import tensorflow as tf
from tensorflow.contrib.image import transform as H_transform
from math import pi

from .backbones.vgg import vgg_block
from superpoint.utils.tools import dict_update


def detector_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'activation': tf.nn.relu, 'batch_normalization': True,
                   'training': config['training']}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    with tf.variable_scope('detector', reuse=tf.AUTO_REUSE):
        x = vgg_block(inputs, 256, 3, 'conv1', **params_conv)
        x = vgg_block(inputs, 1+pow(config['grid_size'], 2), 1, 'conv2', **params_conv)

        prob = tf.nn.softmax(x, axis=cindex)
        # Strip the extra “no interest point” dustbin
        prob = prob[:, :-1, :, :] if cfirst else prob[:, :, :, :-1]
        prob = tf.depth_to_space(
                prob, config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')
        prob = tf.squeeze(prob, axis=cindex)

    return {'logits': x, 'prob': prob}


def spatial_nms(prob, size):
    """Performs non maximum suppression on the heatmap using max-pooling. This method is
    faster than box_nms, but does not suppress contiguous that have the same probability
    value.

    Arguments:
        prob: the probability heatmap, with shape `[H, W]`.
        size: a scalar, the size of the pooling window.
    """

    with tf.name_scope('spatial_nms'):
        prob = tf.expand_dims(tf.expand_dims(prob, axis=0), axis=-1)
        pooled = tf.nn.max_pool(
                prob, ksize=[1, size, size, 1], strides=[1, 1, 1, 1], padding='SAME')
        prob = tf.where(tf.equal(prob, pooled), prob, tf.zeros_like(prob))
        return tf.squeeze(prob)


def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    """Performs non maximum suppression on the heatmap by considering hypothetical
    bounding boxes centered at each pixel's location (e.g. corresponding to the receptive
    field). Optionally only keeps the top k detections.

    Arguments:
        prob: the probability heatmap, with shape `[H, W]`.
        size: a scalar, the size of the bouding boxes.
        iou: a scalar, the IoU overlap threshold.
        min_prob: a threshold under which all probabilities are discarded before NMS.
        keep_top_k: an integer, the number of top scores to keep.
    """
    with tf.name_scope('box_nms'):
        pts = tf.to_float(tf.where(tf.greater_equal(prob, min_prob)))
        size = tf.constant(size/2.)
        boxes = tf.concat([pts-size, pts+size], axis=1)
        scores = tf.gather_nd(prob, tf.to_int32(pts))
        with tf.device('/cpu:0'):
            indices = tf.image.non_max_suppression(
                    boxes, scores, tf.shape(boxes)[0], iou)
        pts = tf.gather(pts, indices)
        scores = tf.gather(scores, indices)
        if keep_top_k:
            k = tf.minimum(tf.shape(scores)[0], tf.constant(keep_top_k))  # when fewer
            scores, indices = tf.nn.top_k(scores, k)
            pts = tf.gather(pts, indices)
        prob = tf.scatter_nd(tf.to_int32(pts), scores, tf.shape(prob))
    return prob


homography_adaptation_default_config = {
        'num': 1,
        'aggregation': 'sum',
        'homographies': {
            'translation': True,
            'rotation': True,
            'scaling': True,
            'perspective': True,
            'scaling_amplitude': 0.1,
            'perspective_amplitude': 0.05,
        },
        'filter_counts': 0
}


def descriptor_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'activation': tf.nn.relu, 'batch_normalization': True,
                   'training': config['training']}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    with tf.variable_scope('descriptor', reuse=tf.AUTO_REUSE):
        x = vgg_block(inputs, 256, 3, 'conv1', **params_conv)
        x = vgg_block(inputs, config['descriptor_size'], 1, 'conv2', **params_conv)

        if cindex == -1:
            with tf.device('/cpu:0'):
                desc = tf.image.resize_bicubic(x,
                                               config['grid_size'] *
                                               tf.shape(x)[1:3])
        else:  # cindex == 1
            # resize_bicubic only supports channels last
            desc = tf.transpose(x, [0, 2, 3, 1])
            with tf.device('/cpu:0'):
                desc = tf.image.resize_bicubic(x,
                                               config['grid_size'] *
                                               tf.shape(desc)[1:3])
            desc = tf.transpose(desc, [0, 3, 1, 2])
        desc = tf.nn.l2_normalize(desc, cindex)

    return {'logits': x, 'descriptor': desc}


def homography_adaptation(image, net, config):
    """Perfoms homography adaptation.

    Inference using multiple random wrapped patches of the same input image for robust
    predictions.

    Arguments:
        image: A `Tensor` with shape `[H, W, 1]`.
        net: A function that takes an image as input, performs inference, and outputs the
            prediction dictionary.
        config: A configuration dictionary containing optional entries such as the number
            of sampled homographies `'num'`, the aggregation method `'aggregation'`.

    Returns:
        A dictionary which contains the aggregated detection probabilities.
    """

    probs = net(image)['prob']
    counts = tf.ones_like(probs)
    images = image

    probs = tf.expand_dims(probs, axis=0)
    counts = tf.expand_dims(counts, axis=0)
    images = tf.expand_dims(images, axis=0)

    shape = tf.shape(image)[:2]
    config = dict_update(homography_adaptation_default_config, config)

    def step(i, probs, counts, images):
        # Sample image patch
        H = sample_homography(shape, **config['homographies'])
        H_inv = invert_homography(H)
        wrapped = H_transform(image, H, interpolation='BILINEAR')
        count = H_transform(tf.ones(shape), H_inv, interpolation='NEAREST')

        # Predict detection probabilities
        input_wrapped = tf.image.resize_images(wrapped, tf.floordiv(shape, 2))
        prob = net(input_wrapped)['prob']
        prob = tf.image.resize_images(tf.expand_dims(prob, axis=-1), shape)[..., 0]

        # Select the points to be mapped back to the original image
        pts = tf.where(tf.greater_equal(prob, 0.01))
        selected_prob = tf.gather_nd(prob, pts)

        # Compute the projected coordinates
        pad = tf.ones(tf.stack([tf.shape(pts)[0], tf.constant(1)]))
        pts_homogeneous = tf.concat([tf.reverse(tf.to_float(pts), axis=[1]), pad], 1)
        pts_proj = tf.matmul(pts_homogeneous, tf.transpose(flat2mat(H)[0]))
        pts_proj = pts_proj[:, :2] / tf.expand_dims(pts_proj[:, 2], axis=1)
        pts_proj = tf.to_int32(tf.round(tf.reverse(pts_proj, axis=[1])))

        # Hack: convert 2D coordinates to 1D indices in order to use tf.unique
        pts_idx = pts_proj[:, 0] * shape[1] + pts_proj[:, 1]
        pts_idx_unique, idx = tf.unique(pts_idx)

        # Keep maximum corresponding probability for each projected point
        # Hack: tf.segment_max requires sorted indices
        idx, sort_idx = tf.nn.top_k(idx, k=tf.shape(idx)[0])
        idx = tf.reverse(idx, axis=[0])
        sort_idx = tf.reverse(sort_idx, axis=[0])
        selected_prob = tf.gather(selected_prob, sort_idx)
        with tf.device('/cpu:0'):
            unique_prob = tf.segment_max(selected_prob, idx)

        # Create final probability map
        pts_proj_unique = tf.stack([tf.floordiv(pts_idx_unique, shape[1]),
                                    tf.floormod(pts_idx_unique, shape[1])], axis=1)
        prob_proj = tf.scatter_nd(pts_proj_unique, unique_prob, shape)

        probs = tf.concat([probs, tf.expand_dims(prob_proj, 0)], axis=0)
        counts = tf.concat([counts, tf.expand_dims(count, 0)], axis=0)
        images = tf.concat([images, tf.expand_dims(wrapped, 0)], axis=0)
        return i + 1, probs, counts, images

    _, probs, counts, images = tf.while_loop(
            lambda i, p, c, im: tf.less(i, config['num'] - 1),
            step,
            [0, probs, counts, images],
            parallel_iterations=1,
            shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([None, None, None]),
                    tf.TensorShape([None, None, None]),
                    tf.TensorShape([None, None, None, 1])])

    counts = tf.reduce_sum(counts, axis=0)
    max_prob = tf.reduce_max(probs, axis=0)
    mean_prob = tf.reduce_sum(probs, axis=0) / counts

    if config['aggregation'] == 'max':
        prob = max_prob
    elif config['aggregation'] == 'sum':
        prob = mean_prob
    else:
        raise ValueError('Unkown aggregation method: {}'.format(config['aggregation']))

    if config['filter_counts']:
        prob = tf.where(tf.greater_equal(counts, config['filter_counts']),
                        prob, tf.zeros_like(prob))

    return {'prob': prob, 'counts': counts,
            'mean_prob': mean_prob, 'input_images': images, 'H_probs': probs}  # debug


def homography_adaptation_batch(images, net, config):
    ha_dtype = {i: tf.float32
                for i in ['prob', 'counts', 'mean_prob', 'input_images', 'H_probs']}

    def net_single(image):
        outputs = net(tf.expand_dims(image, axis=0))
        return {k: v[0] for k, v in outputs.items()}

    return tf.map_fn(lambda image: homography_adaptation(image, net_single, config),
                     images, dtype=ha_dtype)


def sample_homography(
        shape, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=16, scaling_amplitude=0.1, perspective_amplitude=0.1):
    """Sample a random valid homography.

    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: the number of tentative scales that are sampled when scaling.
        n_angles: the number of tentatives angles that are sampled when rotating.

    Returns:
        A `Tensor` of shape `[1, 8]` corresponding to the flattened homography transform.
    """

    # Corners of the output image
    pts1 = tf.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    pts2 = 0.25 + tf.constant([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], tf.float32)

    # Random perspective and affine perturbations
    if perspective:
        pts2 += tf.truncated_normal([4, 2], 0., min(perspective_amplitude, 0.25)/2)

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = tf.concat(
                [[1.], tf.truncated_normal([n_scales], 1, scaling_amplitude/2)], 0)
        center = tf.reduce_mean(pts2, axis=0, keepdims=True)
        scaled = tf.expand_dims(pts2 - center, axis=0) * tf.expand_dims(
                tf.expand_dims(scales, 1), 1) + center
        valid = tf.logical_and(tf.greater_equal(scaled, 0.), tf.less(scaled, 1.))
        valid = tf.where(tf.reduce_all(valid, axis=[1, 2]))
        with tf.device('/cpu:0'):
            idx = tf.random_shuffle(valid)[0]
        pts2 = scaled[idx[0]]

    # Random translation
    if translation:
        t_min, t_max = tf.reduce_min(pts2, axis=0), tf.reduce_min(1 - pts2, axis=0)
        pts2 += tf.expand_dims(tf.stack([tf.random_uniform((), -t_min[0], t_max[0]),
                                         tf.random_uniform((), -t_min[1], t_max[1])]),
                               axis=0)

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = tf.lin_space(0., 2*tf.constant(pi), n_angles)
        center = tf.reduce_mean(pts2, axis=0, keepdims=True)
        rot_mat = tf.reshape(tf.stack([tf.cos(angles), -tf.sin(angles), tf.sin(angles),
                                       tf.cos(angles)], axis=1), [-1, 2, 2])
        rotated = tf.matmul(
                tf.tile(tf.expand_dims(pts2 - center, axis=0), [n_angles, 1, 1]),
                rot_mat) + center
        valid = tf.logical_and(tf.greater_equal(rotated, 0.), tf.less(rotated, 1.))
        valid = tf.where(tf.reduce_all(valid, axis=[1, 2]))
        with tf.device('/cpu:0'):
            idx = tf.random_shuffle(valid)[0]
        pts2 = rotated[idx[0]]

    # Rescale to actual size
    shape = tf.to_float(shape[::-1])  # different convention [y, x]
    pts1 *= tf.expand_dims(shape, axis=0)
    pts2 *= tf.expand_dims(shape, axis=0)

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = tf.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = tf.transpose(tf.stack(
        [[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))
    homography = tf.transpose(tf.matrix_solve_ls(a_mat, p_mat, fast=True))
    return homography


def invert_homography(H):
    """
    Computes the inverse transformation for a flattened homography transformation.
    """
    return mat2flat(tf.matrix_inverse(flat2mat(H)))


def flat2mat(H):
    """
    Converts a flattened homography transformation with shape `[1, 8]` to its
    corresponding homography matrix with shape `[1, 3, 3]`.
    """
    return tf.reshape(tf.concat([H, tf.ones([tf.shape(H)[0], 1])], axis=1), [-1, 3, 3])


def mat2flat(H):
    """
    Converts an homography matrix with shape `[1, 3, 3]` to its corresponding flattened
    homography transformation with shape `[1, 8]`.
    """
    H = tf.reshape(H, [-1, 9])
    return (H / H[:, 8:9])[:, :8]


def warp_keypoints_to_list(packed_arg):
    """
    Warp a map of keypoints (pixel is 1 for a keypoint and 0 else) with
    the INVERSE of the homography H.
    The inverse is used to be coherent with tf.contrib.image.transform

    Arguments:
        packed_arg: a tuple equal to (keypoints_map, H)

    Returns: a Tensor of size (num_keypoints, 2) with the new coordinates
             of the warped keypoints.
    """
    keypoints_map = packed_arg[0]
    H = packed_arg[1]
    if len(H.shape.as_list()) < 2:
        H = tf.expand_dims(H, 0)  # add a batch of 1
    # Get the keypoints list in homogeneous format
    keypoints = tf.cast(tf.where(keypoints_map > 0), tf.float32)
    keypoints = keypoints[:, ::-1]
    n_keypoints = tf.shape(keypoints)[0]
    keypoints = tf.concat([keypoints, tf.ones([n_keypoints, 1], dtype=tf.float32)], 1)

    # Apply the homography
    H_inv = invert_homography(H)
    H_inv = flat2mat(H_inv)
    H_inv = tf.transpose(H_inv[0, ...])
    warped_keypoints = tf.matmul(keypoints, H_inv)
    warped_keypoints = tf.round(warped_keypoints[:, :2]
                                / warped_keypoints[:, 2:])
    warped_keypoints = warped_keypoints[:, ::-1]

    return warped_keypoints


def warp_keypoints_to_map(packed_arg):
    """
    Warp a map of keypoints (pixel is 1 for a keypoint and 0 else) with
    the INVERSE of the homography H.
    The inverse is used to be coherent with tf.contrib.image.transform

    Arguments:
        packed_arg: a tuple equal to (keypoints_map, H)

    Returns: a map of keypoints with of the same format as the original keypoints map.
    """
    warped_keypoints = tf.to_int32(warp_keypoints_to_list(packed_arg))
    n_keypoints = tf.shape(warped_keypoints)[0]
    shape = tf.shape(packed_arg[0])

    # Remove points outside the image
    zeros = tf.cast(tf.zeros([n_keypoints]), dtype=tf.bool)
    ones = tf.cast(tf.ones([n_keypoints]), dtype=tf.bool)
    loc = tf.logical_and(tf.where(warped_keypoints[:, 0] >= 0, ones, zeros),
                         tf.where(warped_keypoints[:, 0] < shape[0],
                                  ones,
                                  zeros))
    loc = tf.logical_and(loc, tf.where(warped_keypoints[:, 1] >= 0, ones, zeros))
    loc = tf.logical_and(loc,
                         tf.where(warped_keypoints[:, 1] < shape[1],
                                  ones,
                                  zeros))
    warped_keypoints = tf.boolean_mask(warped_keypoints, loc)

    # Output the new map of keypoints
    new_map = tf.scatter_nd(warped_keypoints,
                            tf.ones([tf.shape(warped_keypoints)[0]], dtype=tf.float32),
                            shape)

    return new_map


def tf_repeat(tensor, repeats):
    """
    Equivalent of np.repeat but for Tensors.

    Arguments:
        tensor: A Tensor (1-D or higher).
        repeats: A list indicating the number of repeat for each dimension
                 (length must be the same as the number of dimensions in tensor).

    Returns: A Tensor with the same type as tensor and a shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tensor
