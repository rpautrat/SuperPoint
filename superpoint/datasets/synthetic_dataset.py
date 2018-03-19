import cv2 as cv
import numpy as np
import math

""" Module used to generate geometrical synthetic shapes """


def get_random_color(background_color):
    """ Output a random scalar in grayscale with a least a small
        contrast with the background color """
    color = np.random.randint(256)
    if abs(color - background_color) < 30:  # not enough contrast
        color = (color + 128) % 256
    return color


def add_salt_and_pepper(img):
    """ Add salt and pepper noise to an image """
    noise = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv.randu(noise, 0, 255)
    black = noise < 30
    white = noise > 225
    img[white > 0] = 255
    img[black > 0] = 0
    cv.blur(img, (5, 5), img)
    return np.empty((0, 2), dtype=np.int)


def generate_background(size):
    """ Generate a customized background image """
    # img = np.zeros(size, dtype=np.uint8)
    # nb_blobs = 30
    # background_color = np.random.randint(256)
    # img = img + background_color
    # blobs = np.concatenate([np.random.randint(0, size[1], size=(nb_blobs, 1)),
    #                         np.random.randint(0, size[0], size=(nb_blobs, 1))],
    #                        axis=1)
    # for i in range(nb_blobs):
    #     col = get_random_color(background_color)
    #     cv.circle(img, (blobs[i][0], blobs[i][1]),
    #               np.random.randint(20), col, -1)
    # kernel_size = np.random.randint(20, 100)
    # img = cv.blur(img, (kernel_size, kernel_size))
    img = np.zeros(size, dtype=np.uint8)
    cv.randu(img, 0, 255)
    cv.threshold(img, np.random.randint(256), 255, cv.THRESH_BINARY, img)
    cv.blur(img, (20, 20), img)
    return img


def generate_custom_background(size, background_color):
    """ Generate a customized background to fill the shapes """
    # img = np.zeros(size, dtype=np.uint8)
    # nb_blobs = 1000
    # col1 = get_random_color(background_color)
    # img = img + col1
    # blobs = np.concatenate([np.random.randint(0, size[1], size=(nb_blobs, 1)),
    #                         np.random.randint(0, size[0], size=(nb_blobs, 1))],
    #                        axis=1)
    # for i in range(nb_blobs):
    #     col = get_random_color(background_color)
    #     cv.circle(img, (blobs[i][0], blobs[i][1]),
    #               np.random.randint(5), col, -1)
    # kernel_size = np.random.randint(10, 20)
    # img = cv.blur(img, (kernel_size, kernel_size))
    img = np.zeros(size, dtype=np.uint8)
    cv.randu(img, 0, 255)
    cv.threshold(img, np.random.randint(256), 255, cv.THRESH_BINARY, img)
    if abs(np.mean(img) - background_color) < 30:
        img = (img + 128) % 256
    cv.blur(img, (10, 10), img)
    return img


def ccw(A, B, C, dim):
    """ Check if the points are listed in counter-clockwise order """
    if dim == 2:  # only 2 dimensions
        return((C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0])
               > (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0]))
    else:  # dim should be equal to 3
        return((C[:, 1, :] - A[:, 1, :])
               * (B[:, 0, :] - A[:, 0, :])
               > (B[:, 1, :] - A[:, 1, :])
               * (C[:, 0, :] - A[:, 0, :]))


def intersect(A, B, C, D, dim):
    """ Return true if line segments AB and CD intersect """
    return np.any((ccw(A, C, D, dim) != ccw(B, C, D, dim)) &
                  (ccw(A, B, C, dim) != ccw(A, B, D, dim)))


def keep_points_inside(points, size):
    """ Keep only the points whose coordinates are inside the dimensions of
    the image of size 'size' """
    mask = (points[:, 0] >= 0) & (points[:, 0] < size[1]) &\
           (points[:, 1] >= 0) & (points[:, 1] < size[0])
    return points[mask, :]


def draw_lines(img):
    """ Draw up to 10 random lines and output the positions of the endpoints """
    num_lines = np.random.randint(1, 10)
    segments = np.empty((0, 4), dtype=np.int)
    points = np.empty((0, 2), dtype=np.int)
    background_color = int(np.mean(img))
    for i in range(num_lines):
        x1 = np.random.randint(img.shape[1])
        y1 = np.random.randint(img.shape[0])
        p1 = np.array([[x1, y1]])
        x2 = np.random.randint(img.shape[1])
        y2 = np.random.randint(img.shape[0])
        p2 = np.array([[x2, y2]])
        # Check that there is no overlap
        if intersect(segments[:, 0:2], segments[:, 2:4], p1, p2, 2):
            continue
        segments = np.concatenate([segments, np.array([[x1, y1, x2, y2]])], axis=0)
        col = get_random_color(background_color)
        cv.line(img, (x1, y1), (x2, y2), col, np.random.randint(1, 4))
        points = np.concatenate([points, np.array([[x1, y1], [x2, y2]])], axis=0)
    return points


def draw_polygon(img):
    """ Draw a polygon with a random number of corners (between 3 and 5)
    and return the corner points """
    num_corners = np.random.randint(3, 6)
    rad = max(np.random.rand() * min(img.shape[0] / 2, img.shape[1] / 2), 30)
    x = np.random.randint(rad, img.shape[1] - rad)  # Center of a circle
    y = np.random.randint(rad, img.shape[0] - rad)
    # Sample num_corners points inside the circle
    slices = np.linspace(0, 2 * math.pi, num_corners + 1)
    angles = [slices[i] + np.random.rand() * (slices[i+1] - slices[i])
              for i in range(num_corners)]
    points = np.array([[int(x + max(np.random.rand(), 0.5) * rad * math.cos(a)),
                        int(y + max(np.random.rand(), 0.5) * rad * math.sin(a))]
                       for a in angles])
    corners = points.reshape((-1, 1, 2))
    col = get_random_color(int(np.mean(img)))
    cv.fillPoly(img, [corners], col)
    return points


def draw_multiple_polygons(img):
    """ Draw multiple polygons with a random number of corners (between 3 and 5)
    and return the corner points """
    segments = np.empty((0, 4), dtype=np.int)
    points = np.empty((0, 2), dtype=np.int)
    background_color = int(np.mean(img))
    for i in range(30):
        num_corners = np.random.randint(3, 6)
        rad = max(np.random.rand() * min(img.shape[0] / 2, img.shape[1] / 2), 30)
        x = np.random.randint(rad, img.shape[1] - rad)  # Center of a circle
        y = np.random.randint(rad, img.shape[0] - rad)
        # Sample num_corners points inside the circle
        slices = np.linspace(0, 2 * math.pi, num_corners + 1)
        angles = [slices[i] + np.random.rand() * (slices[i+1] - slices[i])
                  for i in range(num_corners)]
        new_points = [[int(x + max(np.random.rand(), 0.5) * rad * math.cos(a)),
                       int(y + max(np.random.rand(), 0.5) * rad * math.sin(a))]
                      for a in angles]
        new_points = np.array(new_points)
        new_segments = np.zeros((1, 4, num_corners))
        new_segments[:, 0, :] = [new_points[i][0] for i in range(num_corners)]
        new_segments[:, 1, :] = [new_points[i][1] for i in range(num_corners)]
        new_segments[:, 2, :] = [new_points[(i+1) % num_corners][0]
                                 for i in range(num_corners)]
        new_segments[:, 3, :] = [new_points[(i+1) % num_corners][1]
                                 for i in range(num_corners)]

        # Check that the polygon will not overlap with pre-existing shapes
        if intersect(segments[:, 0:2, None],
                     segments[:, 2:4, None],
                     new_segments[:, 0:2, :],
                     new_segments[:, 2:4, :],
                     3):
            continue
        new_segments = np.reshape(np.swapaxes(new_segments, 0, 2), (-1, 4))
        segments = np.concatenate([segments, new_segments], axis=0)

        # Color the polygon with a custom background
        corners = new_points.reshape((-1, 1, 2))
        mask = np.zeros(img.shape, np.uint8)
        custom_background = generate_custom_background(img.shape, background_color)
        cv.fillPoly(mask, [corners], 255)
        locs = np.where(mask != 0)
        img[locs[0], locs[1]] = custom_background[locs[0], locs[1]]
        points = np.concatenate([points, new_points], axis=0)
    return points


def draw_ellipses(img):
    """ Draw several ellipses """
    centers = np.empty((0, 2), dtype=np.int)
    rads = np.empty((0, 1), dtype=np.int)
    min_dim = min(img.shape[0], img.shape[1]) / 4
    background_color = int(np.mean(img))
    for i in range(20):  # at most 20 ellipses (if no overlap)
        ax = int(max(np.random.rand() * min_dim, 15))  # semi axis of the ellipse
        ay = int(max(np.random.rand() * min_dim, 15))  # semi axis of the ellipse
        max_rad = max(ax, ay)
        x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
        y = np.random.randint(max_rad, img.shape[0] - max_rad)
        new_center = np.array([[x, y]])

        # Check that the ellipsis will not overlap with pre-existing shapes
        diff = centers - new_center
        if np.any(max_rad > (np.sqrt(np.sum(diff * diff, axis=1)) - rads)):
            continue
        centers = np.concatenate([centers, new_center], axis=0)
        rads = np.concatenate([rads, np.array([[max_rad]])], axis=0)

        col = get_random_color(background_color)
        angle = np.random.rand() * 90
        cv.ellipse(img, (x, y), (ax, ay), angle, 0, 360, col, -1)
    return np.empty((0, 2), dtype=np.int)


def draw_star(img):
    """ Draw a star with between 3 and 5 branches and output the interest points """
    num_branches = np.random.randint(3, 6)
    thickness = np.random.randint(1, 3)
    rad = max(np.random.rand() * min(img.shape[0] / 2, img.shape[1] / 2), 50)
    x = np.random.randint(rad, img.shape[1] - rad)  # select the center of a circle
    y = np.random.randint(rad, img.shape[0] - rad)
    # Sample num_branches points inside the circle
    slices = np.linspace(0, 2 * math.pi, num_branches + 1)
    angles = [slices[i] + np.random.rand() * (slices[i+1] - slices[i])
              for i in range(num_branches)]
    points = np.array([[int(x + max(np.random.rand(), 0.3) * rad * math.cos(a)),
                        int(y + max(np.random.rand(), 0.3) * rad * math.sin(a))]
                       for a in angles])
    points = np.concatenate(([[x, y]], points), axis=0)
    background_color = int(np.mean(img))
    for i in range(1, num_branches + 1):
        col = get_random_color(background_color)
        cv.line(img, (points[0][0], points[0][1]),
                (points[i][0], points[i][1]),
                col, thickness)
    return points


def draw_checkerboard(img):
    """ Draw a checkerboard and output the interest points """
    background_color = int(np.mean(img))
    # Create the grid
    rows = np.random.randint(3, 7)  # number of rows
    cols = np.random.randint(3, 7)  # number of cols
    s = min((img.shape[1] - 1) // cols, (img.shape[0] - 1) // rows)  # size of a cell
    board = np.zeros((rows * s, cols * s), np.uint8)
    x_coord = np.tile(range(cols + 1),
                      rows + 1).reshape(((rows + 1) * (cols + 1), 1))
    y_coord = np.repeat(range(rows + 1),
                        cols + 1).reshape(((rows + 1) * (cols + 1), 1))
    points = s * np.concatenate([x_coord, y_coord], axis=1)

    # Fill the rectangles
    for i in range(rows):
        for j in range(cols):
            col = get_random_color(background_color)
            cv.rectangle(board, (j * s, i * s), ((j + 1) * s, (i + 1) * s), col, -1)
    # cv.imshow("Checkerboard", board)

    # Warp the grid using an affine transformation
    # The parameters of the affine transformation are a bit constrained
    # to get transformations not too far-fetched
    scale = 0.5 + np.random.rand() * 1.5
    angle = np.random.rand() * 2 * math.pi
    affine_transform = [[scale * max(np.random.rand(), 0.6) * math.cos(angle),
                         scale * min(np.random.rand(), 0.4) * math.sin(angle),
                         rows * s / 2 + np.random.randint(-30, 30)],
                        [- scale * min(np.random.rand(), 0.4) * math.sin(angle),
                         scale * max(np.random.rand(), 0.6) * math.cos(angle),
                         cols * s / 2 + np.random.randint(-30, 30)]]
    affine_transform = np.array(affine_transform)
    warped_board = np.transpose(cv.warpAffine(board,
                                              affine_transform,
                                              img.shape[0:2]))
    points = np.transpose(np.concatenate((points,
                                          np.ones(((rows + 1) * (cols + 1), 1))),
                                         axis=1))
    warped_points = np.transpose(np.dot(affine_transform, points))
    warped_points = warped_points.astype(int)
    warped_points[:, [0, 1]] = warped_points[:, [1, 0]]  # x and y have been inverted
    # cv.imshow("Warped checkerboard", warped_board)

    # Add the warped checkerboard to img
    mask = warped_board
    locs = np.where(mask != 0)
    img[locs[0], locs[1]] = mask[locs[0], locs[1]]

    # Keep only the points inside the image
    points = keep_points_inside(warped_points, img.shape[:2])
    return points


def draw_stripes(img):
    """ Draw stripes in a distorted rectangle and output the interest points """
    background_color = int(np.mean(img))
    # Create the grid
    board_size = (int(img.shape[0] * (1 + np.random.rand())),
                  int(img.shape[1] * (1 + np.random.rand())))
    board = np.zeros(board_size, np.uint8)
    col = np.random.randint(5, 13)  # number of cols
    cols = np.concatenate([board_size[1] * np.random.rand(col - 1),
                           np.array([0, board_size[1] - 1])], axis=0)
    cols = np.unique(cols.astype(int))
    # Remove the indices that are too close
    cols = cols[(np.concatenate([cols[1:],
                                 np.array([board_size[1] + 10])],
                                axis=0) - cols) >= 10]
    col = cols.shape[0] - 1  # update the number of cols
    cols = np.reshape(cols, (col + 1, 1))
    cols1 = np.concatenate([cols, np.zeros((col + 1, 1), np.int32)], axis=1)
    cols2 = np.concatenate([cols,
                            (board_size[0] - 1) * np.ones((col + 1, 1), np.int32)],
                           axis=1)
    points = np.concatenate([cols1, cols2], axis=0)

    # Fill the rectangles
    color = get_random_color(background_color)
    for i in range(col):
        color = (color + 128 + np.random.randint(-30, 30)) % 256
        cv.rectangle(board, (points[i][0], points[i][1]),
                     (points[i+col+2][0], points[i+col+2][1]),
                     color, -1)
    # cv.imshow("Stripes", board)

    # Warp the grid using an affine transformation
    # The parameters of the affine transformation are a bit constrained
    # to get transformations not too far-fetched
    scale = 0.6 + np.random.rand() * 0.4
    angle = np.random.rand() * 2 * math.pi
    affine_transform = [[scale * max(np.random.rand(), 0.7) * math.cos(angle),
                         scale * min(np.random.rand(), 0.4) * math.sin(angle),
                         board_size[0] / 3 + np.random.randint(-10, 10)],
                        [- scale * min(np.random.rand(), 0.4) * math.sin(angle),
                         scale * max(np.random.rand(), 0.7) * math.cos(angle),
                         board_size[1] / 3 + np.random.randint(-10, 10)]]
    affine_transform = np.array(affine_transform)
    trans = np.array([[1, 0, - board_size[0] / 2],
                      [0, 1, - board_size[1] / 2],
                      [0, 0, 1]])
    affine_transform = np.dot(affine_transform, trans)
    warped_board = np.transpose(cv.warpAffine(board,
                                              affine_transform,
                                              img.shape[0:2]))
    points = np.transpose(np.concatenate((points,
                                          np.ones((2 * (col + 1), 1))),
                                         axis=1))
    warped_points = np.transpose(np.dot(affine_transform, points))
    warped_points = warped_points.astype(int)
    warped_points[:, [0, 1]] = warped_points[:, [1, 0]]  # x and y have been inverted
    # cv.imshow("Warped stripes", warped_board)

    # Add the warped stripes to img
    mask = warped_board
    locs = np.where(mask != 0)
    img[locs[0], locs[1]] = mask[locs[0], locs[1]]

    # Keep only the points inside the image
    points = keep_points_inside(warped_points, img.shape[:2])
    return points


def draw_cube(img):
    """ Draw a 2D projection of a cube and output the corners that are visible """
    # Generate a cube and apply to it an affine transformation
    # The order matters!
    # The indices of two adjacent vertices differ only of one bit (as in Gray codes)
    background_color = int(np.mean(img))
    min_dim = min(img.shape[:2])
    lx = 30 + np.random.rand() * min_dim / 2  # dimensions of the cube
    ly = 30 + np.random.rand() * min_dim / 2
    lz = 30 + np.random.rand() * min_dim / 2
    cube = np.array([[0, 0, 0],
                     [lx, 0, 0],
                     [0, ly, 0],
                     [lx, ly, 0],
                     [0, 0, lz],
                     [lx, 0, lz],
                     [0, ly, lz],
                     [lx, ly, lz]])
    rot_angles = np.random.rand(3) * 3 * math.pi / 10. + math.pi / 10.
    rotation_1 = np.array([[math.cos(rot_angles[0]), -math.sin(rot_angles[0]), 0],
                           [math.sin(rot_angles[0]), math.cos(rot_angles[0]), 0],
                           [0, 0, 1]])
    rotation_2 = np.array([[1, 0, 0],
                           [0, math.cos(rot_angles[1]), -math.sin(rot_angles[1])],
                           [0, math.sin(rot_angles[1]), math.cos(rot_angles[1])]])
    rotation_3 = np.array([[math.cos(rot_angles[2]), 0, -math.sin(rot_angles[2])],
                           [0, 1, 0],
                           [math.sin(rot_angles[2]), 0, math.cos(rot_angles[2])]])
    scaling = np.array([[0.4 + np.random.rand() * 0.6, 0, 0],
                        [0, 0.4 + np.random.rand() * 0.6, 0],
                        [0, 0, 0.4 + np.random.rand() * 0.6]])
    trans = np.array([img.shape[1] / 2 + np.random.randint(-img.shape[1] / 5,
                                                           img.shape[1] / 5),
                      img.shape[0] / 2 + np.random.randint(-img.shape[0] / 5,
                                                           img.shape[0] / 5),
                      0])
    cube = trans + np.transpose(np.dot(scaling,
                                       np.dot(rotation_1,
                                              np.dot(rotation_2,
                                                     np.dot(rotation_3,
                                                            np.transpose(cube))))))

    # The hidden corner is 0 by construction
    # The front one is 7
    cube = cube[:, :2]  # project on the plane z=0
    cube = cube.astype(int)
    points = cube[1:, :]  # get rid of the hidden corner

    # Get the three visible faces
    faces = np.array([[7, 3, 1, 5], [7, 5, 4, 6], [7, 6, 2, 3]])

    # Fill the faces and draw the contours
    col_face = get_random_color(background_color)
    for i in [0, 1, 2]:
        cv.fillPoly(img, [cube[faces[i]].reshape((-1, 1, 2))],
                    col_face)
    thickness = np.random.randint(1, 3)
    for i in [0, 1, 2]:
        for j in [0, 1, 2, 3]:
            col_edge = (col_face + 128
                        + np.random.randint(-64, 64))\
                        % 256  # color that constrats with the face color
            cv.line(img, (cube[faces[i][j], 0], cube[faces[i][j], 1]),
                    (cube[faces[i][(j + 1) % 4], 0], cube[faces[i][(j + 1) % 4], 1]),
                    col_edge, thickness)

    # Keep only the points inside the image
    points = keep_points_inside(points, img.shape[:2])
    return points


def gaussian_noise(img):
    """ Apply random noise to the image """
    cv.randu(img, 0, 255)
    return np.empty((0, 2), dtype=np.int)


def draw_shape(img):
    """ Draw a shape randomly """
    idx = np.random.randint(8)
    if idx == 0:
        return draw_lines(img)
    if idx == 1:
        return draw_multiple_polygons(img)
    if idx == 2:
        return draw_ellipses(img)
    if idx == 3:
        return draw_star(img)
    if idx == 4:
        return draw_checkerboard(img)
    if idx == 5:
        return gaussian_noise(img)
    if idx == 6:
        return draw_stripes(img)
    else:
        return draw_cube(img)


def draw_interest_points(img, points):
    """ Convert img in RGB and draw in green the interest points """
    img_rgb = np.stack([img, img, img], axis=2)
    for i in range(points.shape[0]):
        cv.circle(img_rgb, (points[i][0], points[i][1]), 2, (0, 255, 0), 1)
    return img_rgb
