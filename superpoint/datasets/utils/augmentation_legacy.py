import cv2 as cv
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter

""" Data augmentation of 2D images """

augmentations = [
        'additive_gaussian_noise',
        'additive_speckle_noise',
        'random_brightness',
        'random_contrast',
        'affine_transform',
        'perspective_transform',
        'elastic_transform',
        'random_crop',
        'add_shade',
        'motion_blur'
]


def dummy(image, keypoints):
    return image, keypoints


def keep_points_inside(points, size):
    """ Keep only the points whose coordinates are inside the dimensions of
    the image of size 'size' """
    mask = (points[:, 0] >= 0) & (points[:, 0] <= (size[1]-1)) &\
           (points[:, 1] >= 0) & (points[:, 1] <= (size[0]-1))
    return points[mask, :]


def additive_gaussian_noise(img, keypoints, random_state=None, std=(5, 95)):
    """ Add gaussian noise to the current image pixel-wise
    Parameters:
      std: the standard deviation of the filter will be between std[0] and std[0]+std[1]
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    sigma = std[0] + random_state.rand() * std[1]
    gaussian_noise = random_state.randn(*img.shape) * sigma
    noisy_img = img + gaussian_noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return (noisy_img, keypoints)


def additive_speckle_noise(img, keypoints, intensity=5):
    """ Add salt and pepper noise to an image
    Parameters:
      intensity: the higher, the more speckles there will be
    """
    noise = np.zeros(img.shape, dtype=np.uint8)
    cv.randu(noise, 0, 256)
    black = noise < intensity
    white = noise > 255 - intensity
    noisy_img = img.copy()
    noisy_img[white > 0] = 255
    noisy_img[black > 0] = 0
    return (noisy_img, keypoints)


def random_brightness(img, keypoints, random_state=None, max_change=50):
    """ Change the brightness of img
    Parameters:
      max_change: max amount of brightness added/subtracted to the image
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    brightness = random_state.randint(-max_change, max_change)
    new_img = img.astype(np.int16) + brightness
    return (np.clip(new_img, 0, 255), keypoints)


def random_contrast(img, keypoints, random_state=None, max_change=[0.5, 1.5]):
    """ Change the contrast of img
    Parameters:
      max_change: the change in contrast will be between 1-max_change and 1+max_change
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    contrast = random_state.uniform(*max_change)
    mean = np.mean(img, axis=(0, 1))
    new_img = np.clip(mean + (img - mean) * contrast, 0, 255)
    return (new_img.astype(np.uint8), keypoints)


def resize_after_crop(orig_img, cropped_img, keypoints, random_state=None):
    """ Crop cropped_img so that it has the same ratio as orig_img
    and resize it to orig_img.shape """
    shape = orig_img.shape
    ratio = shape[0] / shape[1]  # we want to keep the same ratio
    if (cropped_img.shape[1] * ratio) > cropped_img.shape[0]:  # columns too long
        col_length = int(cropped_img.shape[0] / ratio)
        min_col = cropped_img.shape[1] // 2 - col_length // 2
        max_col = min_col + col_length
        resized_img = cropped_img[:, min_col:(max_col+1)]
        mask = (keypoints[:, 0] >= min_col) & (keypoints[:, 0] <= max_col)
        new_keypoints = keypoints[mask, :].astype(float)
        new_keypoints[:, 0] -= min_col
        ratio_x = shape[1] / col_length
        ratio_y = shape[0] / cropped_img.shape[0]
    else:  # rows too long
        row_length = int(cropped_img.shape[1] * ratio)
        min_row = cropped_img.shape[0] // 2 - row_length // 2
        max_row = min_row + row_length
        resized_img = cropped_img[min_row:(max_row+1), :]
        mask = (keypoints[:, 1] >= min_row) & (keypoints[:, 1] <= max_row)
        new_keypoints = keypoints[mask, :].astype(float)
        new_keypoints[:, 1] -= min_row
        ratio_x = shape[1] / cropped_img.shape[1]
        ratio_y = shape[0] / row_length

    # Resize the keypoints
    new_keypoints[:, 0] *= ratio_x
    new_keypoints[:, 1] *= ratio_y
    new_keypoints = keep_points_inside(new_keypoints, shape)
    return (cv.resize(resized_img, (shape[1], shape[0])), new_keypoints)


def crop_after_transform(orig_img, warped_img, transform,
                         orig_keypoints, keypoints, random_state=None):
    """ Crop img after transform has been applied """
    shape = warped_img.shape
    # Compute the new location of the corners
    corners = np.array([[0, 0, 1],
                        [0, shape[0] - 1, 1],
                        [shape[1] - 1, 0, 1],
                        [shape[1] - 1, shape[0] - 1, 1]])
    corners = np.transpose(corners)
    corners = np.dot(transform, corners)
    corners = np.transpose(corners)
    if corners.shape[1] == 3:  # transform is an homography
        corners = corners[:, :2] / corners[:, 2].reshape((4, 1))

    # Crop and resize img
    min_row = max([0, corners[0, 1], corners[2, 1]])
    max_row = min([shape[0], corners[1, 1] + 1, corners[3, 1] + 1])
    min_col = max([0, corners[0, 0], corners[1, 0]])
    max_col = min([shape[1], corners[2, 0] + 1, corners[3, 0] + 1])
    if max_row < min_row + 50 or max_col < min_col + 50:  # retry if too small
        if transform.shape[0] == 2:  # affine transform
            return affine_transform(orig_img, orig_keypoints, random_state)
        else:  # homography
            return perspective_transform(orig_img, orig_keypoints, random_state)
    cropped_img = warped_img[int(min_row):int(max_row), int(min_col):int(max_col)]

    # Crop the keypoints
    mask = (keypoints[:, 0] >= min_col) & (keypoints[:, 0] < max_col) &\
           (keypoints[:, 1] >= min_row) & (keypoints[:, 1] < max_row)
    keypoints = keypoints[mask, :]
    keypoints[:, 0] -= min_col
    keypoints[:, 1] -= min_row
    return resize_after_crop(orig_img, cropped_img, keypoints, random_state)


def affine_transform(img, keypoints, random_state=None, affine_params=(0.05, 0.15)):
    """ Apply an affine transformation to the image
    Parameters:
      affine_params: parameters to modify the affine transformation
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = img.shape
    alpha_affine = np.max(shape) * (affine_params[0] +
                                    random_state.rand() * affine_params[1])
    center_square = np.float32(shape) // 2
    square_size = min(shape) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0]+square_size, center_square[1]-square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine,
                                       alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    M = cv.getAffineTransform(pts1, pts2)

    # Warp the image and keypoints
    warped_img = cv.warpAffine(img, M, shape[::-1])
    new_keypoints = np.transpose(np.concatenate([keypoints,
                                                 np.ones((keypoints.shape[0], 1))],
                                                axis=1))
    new_keypoints = np.transpose(np.dot(M, new_keypoints))

    return crop_after_transform(img, warped_img, M, keypoints,
                                new_keypoints, random_state)


def perspective_transform(img, keypoints, random_state=None, param=0.002):
    """ Apply a perspective transformation to the image
    Parameters:
      param: parameter controlling the intensity of the perspective transform
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    perspective_transform = np.array([[1 - param + 2 * param * random_state.rand(),
                                       -param + 2 * param * random_state.rand(),
                                       -param + 2 * param * random_state.rand()],
                                      [-param + 2 * param * random_state.rand(),
                                       1 - param + 2 * param * random_state.rand(),
                                       -param + 2 * param * random_state.rand()],
                                      [-param + 2 * param * random_state.rand(),
                                       -param + 2 * param * random_state.rand(),
                                       1 - param + 2 * param * random_state.rand()]])

    # Warp the image and the keypoints
    warped_img = cv.warpPerspective(img, perspective_transform, img.shape[::-1])
    warped_col0 = np.add(np.sum(np.multiply(keypoints,
                                            perspective_transform[0, :2]), axis=1),
                         perspective_transform[0, 2])
    warped_col1 = np.add(np.sum(np.multiply(keypoints,
                                            perspective_transform[1, :2]), axis=1),
                         perspective_transform[1, 2])
    warped_col2 = np.add(np.sum(np.multiply(keypoints,
                                            perspective_transform[2, :2]), axis=1),
                         perspective_transform[2, 2])
    warped_col0 = np.divide(warped_col0, warped_col2)
    warped_col1 = np.divide(warped_col1, warped_col2)
    new_keypoints = np.concatenate([warped_col0[:, None], warped_col1[:, None]], axis=1)
    return crop_after_transform(img, warped_img, perspective_transform,
                                keypoints, new_keypoints, random_state)


def elastic_transform(img, keypoints, random_state=None,
                      sigma_params=(0.05, 0.05), alpha_params=(1, 5), padding=10):
    """ Apply an elastic distortion to the image
    Parameters:
      sigma_params: sigma can vary between max(img.shape) * sigma_params[0] and
                    max(img.shape) * (sigma_params[0] + sigma_params[1])
      alpha_params: alpha can vary between max(img.shape) * alpha_params[0] and
                    max(img.shape) * (alpha_params[0] + alpha_params[1])
      padding: padding that will be removed when cropping (remove strange artefacts)
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = img.shape
    sigma = np.max(shape) * (sigma_params[0] + sigma_params[1] * random_state.rand())
    alpha = np.max(shape) * (alpha_params[0] + alpha_params[1] * random_state.rand())

    # Create the grid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    # Apply the distortion
    distorted_img = cv.remap(img, np.float32(x + dx), np.float32(y + dy),
                             interpolation=cv.INTER_LINEAR)
    inverse_map_x = np.float32(x - dx)
    inverse_map_y = np.float32(y - dy)
    keypoints_x = inverse_map_x[keypoints[:, 1], keypoints[:, 0]]
    keypoints_y = inverse_map_y[keypoints[:, 1], keypoints[:, 0]]
    new_keypoints = np.concatenate([keypoints_x[:, None], keypoints_y[:, None]], axis=1)

    # Crop and resize
    min_row = int(math.ceil(np.max(dy))) + padding
    max_row = int(math.floor(np.min(dy))) + shape[0] - padding
    min_col = int(math.ceil(np.max(dx))) + padding
    max_col = int(math.floor(np.min(dx))) + shape[1] - padding
    distorted_img = distorted_img[min_row:max_row, min_col:max_col]
    mask = (new_keypoints[:, 0] >= min_col) & (new_keypoints[:, 0] < max_col) &\
           (new_keypoints[:, 1] >= min_row) & (new_keypoints[:, 1] < max_row)
    new_keypoints = new_keypoints[mask, :]
    new_keypoints[:, 0] -= min_col
    new_keypoints[:, 1] -= min_row
    return resize_after_crop(img, distorted_img, new_keypoints, random_state)


def random_crop(img, keypoints, random_state=None, min_crop_ratio=0.5):
    """ Crop a part of the image and resize to the original size
    Parameters:
      min_crop_ratio: the cropped image will have
                      at least a size min_crop_ratio * img.shape
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = img.shape
    ratio = shape[0] / shape[1]
    new_col_length = random_state.randint(min_crop_ratio * shape[1], shape[1])
    new_row_length = int(ratio * new_col_length)
    start_col = random_state.randint(shape[1] - new_col_length)
    start_row = random_state.randint(shape[0] - new_row_length)
    cropped_img = img[start_row:(start_row+new_row_length),
                      start_col:(start_col+new_col_length)]
    mask = (keypoints[:, 0] >= start_col) &\
           (keypoints[:, 0] < start_col+new_col_length) &\
           (keypoints[:, 1] >= start_row) &\
           (keypoints[:, 1] < start_row+new_row_length)
    new_keypoints = keypoints[mask, :].astype(float)

    # Resize the keypoints and the image
    ratio_x = shape[1] / new_col_length
    ratio_y = shape[0] / new_row_length
    new_keypoints[:, 0] = (new_keypoints[:, 0] - start_col) * ratio_x
    new_keypoints[:, 1] = (new_keypoints[:, 1] - start_row) * ratio_y
    return (cv.resize(cropped_img, (shape[1], shape[0])), new_keypoints)


def add_shade(img, keypoints, random_state=None, nb_ellipses=20,
              amplitude=[-0.5, 0.8], kernel_size_interval=(250, 350)):
    """ Overlay the image with several shades
    Parameters:
      nb_ellipses: number of shades
      amplitude: tuple containing the illumination bound (between -1 and 0) and the
        shawdow bound (between 0 and 1)
      kernel_size_interval: interval of the kernel used to blur the shades
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    transparency = random_state.uniform(*amplitude)

    min_dim = min(img.shape) / 4
    mask = np.zeros(img.shape[:2], np.uint8)
    for i in range(nb_ellipses):
        ax = int(max(random_state.rand() * min_dim, min_dim / 5))
        ay = int(max(random_state.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = random_state.randint(max_rad, img.shape[1] - max_rad)  # center
        y = random_state.randint(max_rad, img.shape[0] - max_rad)
        angle = random_state.rand() * 90
        cv.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

    kernel_size = int(kernel_size_interval[0] + random_state.rand() *
                      (kernel_size_interval[1] - kernel_size_interval[0]))
    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1
    mask = cv.GaussianBlur(mask.astype(np.float), (kernel_size, kernel_size), 0)
    shaded = img * (1 - transparency * mask/255.)
    shaded = np.clip(shaded, 0, 255)
    return (shaded.astype(np.uint8), keypoints)


def add_fog(img, keypoints, random_state=None, max_nb_ellipses=20,
            transparency=0.4, kernel_size_interval=(150, 250)):
    """ Overlay the image with several shades
    Parameters:
      max_nb_ellipses: number max of shades
      transparency: level of transparency of the shades (1 = no shade)
      kernel_size_interval: interval of the kernel used to blur the shades
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    centers = np.empty((0, 2), dtype=np.int)
    rads = np.empty((0, 1), dtype=np.int)
    min_dim = min(img.shape) / 4
    shaded_img = img.copy()
    for i in range(max_nb_ellipses):
        ax = int(max(random_state.rand() * min_dim, min_dim / 5))
        ay = int(max(random_state.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = random_state.randint(max_rad, img.shape[1] - max_rad)  # center
        y = random_state.randint(max_rad, img.shape[0] - max_rad)
        new_center = np.array([[x, y]])

        # Check that the ellipsis will not overlap with pre-existing shapes
        diff = centers - new_center
        if np.any(max_rad > (np.sqrt(np.sum(diff * diff, axis=1)) - rads)):
            continue
        centers = np.concatenate([centers, new_center], axis=0)
        rads = np.concatenate([rads, np.array([[max_rad]])], axis=0)

        col = random_state.randint(256)  # color of the shade
        angle = random_state.rand() * 90
        cv.ellipse(shaded_img, (x, y), (ax, ay), angle, 0, 360, col, -1)
    shaded_img = shaded_img.astype(float)
    kernel_size = int(kernel_size_interval[0] + random_state.rand() *
                      (kernel_size_interval[1] - kernel_size_interval[0]))
    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1

    cv.GaussianBlur(shaded_img, (kernel_size, kernel_size), 0, shaded_img)
    mask = np.where(shaded_img != img)
    shaded_img[mask] = (1 - transparency) * shaded_img[mask] + transparency * img[mask]
    shaded_img = np.clip(shaded_img, 0, 255)
    return (shaded_img.astype(np.uint8), keypoints)


def motion_blur(img, keypoints, max_ksize=10):
    # Either vertial, hozirontal or diagonal blur
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_ksize+1)/2)*2 + 1  # make sure is odd
    center = int((ksize-1)/2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    img = cv.filter2D(img.astype(np.uint8), -1, kernel)
    return img, keypoints
