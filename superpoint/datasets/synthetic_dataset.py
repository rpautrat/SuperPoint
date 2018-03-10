import cv2 as cv
import numpy as np
import math
from scipy.spatial import ConvexHull
from superpoint.utils.bitset import Bitset


class SyntheticDataset():
    """ Methods for generating the synthetic dataset """

    def get_random_color(self):
        """ Output a random scalar in grayscale """
        return np.random.randint(256)

    def add_salt_and_pepper(self, img):
        """ Add salt and pepper noise to an image """
        noise = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv.randu(noise, 0, 255)
        black = noise < 30
        white = noise > 225
        noisy_img = img.copy()
        noisy_img[white > 0] = 255
        noisy_img[black > 0] = 0
        noisy_img = cv.blur(noisy_img, (5, 5))
        return noisy_img

    def generate_background(self, size):
        """ Generate a customized background image """
        img = np.zeros(size, dtype=np.uint8)
        nb_blobs = 100
        col1 = self.get_random_color()
        img = img + col1
        blobs = np.concatenate([np.random.randint(0, size[1], size=(nb_blobs, 1)),
                                np.random.randint(0, size[0], size=(nb_blobs, 1))],
                               axis=1)
        for i in range(nb_blobs):
            col = max(min(col1 + np.random.randint(-100, 100), 255), 0)
            cv.circle(img, (blobs[i][0], blobs[i][1]),
                      np.random.randint(20), (col, col, col), -1)
        kernel_size = np.random.randint(20, 100)
        img = cv.blur(img, (kernel_size, kernel_size))
        return img

    def custom_background(self, size):
        """ Generate a customized background to fill the shapes """
        img = np.zeros(size, dtype=np.uint8)
        nb_blobs = 1000
        col1 = self.get_random_color()
        img = img + col1
        blobs = np.concatenate([np.random.randint(0, size[1], size=(nb_blobs, 1)),
                                np.random.randint(0, size[0], size=(nb_blobs, 1))],
                               axis=1)
        for i in range(nb_blobs):
            col = self.get_random_color()
            cv.circle(img, (blobs[i][0], blobs[i][1]),
                      np.random.randint(5), (col, col, col), -1)
        kernel_size = np.random.randint(10, 20)
        img = cv.blur(img, (kernel_size, kernel_size))
        return img

    def ccw(self, A, B, C):
        """ Check if the points are listed in counter-clockwise order """
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(self, A, B, C, D):
        """ Return true if line segments AB and CD intersect """
        return(self.ccw(A, C, D) != self.ccw(B, C, D) and
               self.ccw(A, B, C) != self.ccw(A, B, D))

    def draw_lines(self, img):
        """ Draw up to 10 random lines and output the positions of the endpoints """
        num_lines = np.random.randint(1, 10)
        segments = []
        points = np.array([[]])
        for i in range(num_lines):
            x1 = np.random.randint(img.shape[1])
            y1 = np.random.randint(img.shape[0])
            x2 = np.random.randint(img.shape[1])
            y2 = np.random.randint(img.shape[0])
            # Check that there is no overlap
            flag = False
            for j in range(len(segments)):
                if self.intersect((x1, y1), (x2, y2), segments[j][0], segments[j][1]):
                    flag = True
                    break
            if flag:
                continue
            segments.append(((x1, y1), (x2, y2)))
            col = self.get_random_color()
            cv.line(img, (x1, y1), (x2, y2), (col, col, col), np.random.randint(1, 4))
            if points.shape == (1, 0):
                points = np.array([[x1, y1], [x2, y2]])
            else:
                points = np.concatenate([points, np.array([[x1, y1], [x2, y2]])], axis=0)
        return points

    def draw_polygon(self, img):
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
        col = self.get_random_color()
        cv.fillPoly(img, [corners], (col, col, col))
        return points

    def draw_multiple_polygons(self, img):
        """ Draw multiple polygons with a random number of corners (between 3 and 5)
        and return the corner points """
        segments = []
        points = np.array([[]])
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
            new_segments = [((new_points[i][0], new_points[i][1]),
                             (new_points[i+1][0], new_points[i+1][1]))
                            for i in range(num_corners - 1)]
            new_segments.append(((new_points[num_corners - 1][0], new_points[num_corners - 1][1]),
                             (new_points[0][0], new_points[0][1])))
            
            # Check that the polygon will not overlap with pre-existing shapes
            flag = False
            for seg in new_segments:
                for prev_seg in segments:
                    if self.intersect(seg[0], seg[1], prev_seg[0], prev_seg[1]):
                        flag = True
                        break
                if flag:
                    break
            if flag:  # there is an overlap
                continue
            segments = segments + new_segments

            # Color the polygon with a custom background
            corners = new_points.reshape((-1, 1, 2))
            mask = np.zeros(img.shape, np.uint8)
            custom_background = self.custom_background(img.shape)
            cv.fillPoly(mask, [corners], (255, 255, 255))
            locs = np.where(mask != 0)
            img[locs[0], locs[1]] = custom_background[locs[0], locs[1]]
            if points.shape == (1, 0):
                points = new_points
            else:
                points = np.concatenate([points, new_points], axis=0)
        return points

    def draw_ellipses(self, img):
        """ Draw several ellipses """
        centers = []
        rads = []
        min_dim = min(img.shape[0], img.shape[1]) / 4
        for i in range(20):  # at most 20 ellipses (if no overlap)
            ax = int(max(np.random.rand() * min_dim, 15))  # semi axis of the ellipse
            ay = int(max(np.random.rand() * min_dim, 15))  # semi axis of the ellipse
            max_rad = max(ax, ay)
            x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center of the ellipse
            y = np.random.randint(max_rad, img.shape[0] - max_rad)
            
            # Check that the ellipsis will not overlap with pre-existing shapes
            flag = False
            for j in range(len(centers)):
                if max_rad + rads[j] > math.sqrt((x - centers[j][0]) ** 2 +
                                             (y - centers[j][1]) ** 2):
                    flag = True
                    break
            if flag:  # there is an overlap
                continue
            centers.append((x, y))
            rads.append(max_rad)
            
            col = self.get_random_color()
            angle = np.random.rand() * 90
            cv.ellipse(img, (x, y), (ax, ay), angle, 0, 360, (col, col, col), -1)
        return np.array([])

    def draw_star(self, img):
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
        for i in range(1, num_branches + 1):
            col = self.get_random_color()
            cv.line(img, (points[0][0], points[0][1]),
                    (points[i][0], points[i][1]),
                    (col, col, col), thickness)
        return points

    def draw_checkerboard(self, img):
        """ Draw a checkerboard and output the interest points """
        # Create the grid
        rows = np.random.randint(3, 7)  # number of rows
        cols = np.random.randint(3, 7)  # number of cols
        s = min((img.shape[1] - 1) // cols, (img.shape[0] - 1) // rows)  # size of a cell
        board = np.zeros((rows * s, cols * s), np.uint8)
        points = np.zeros(((rows + 1) * (cols + 1), 2), np.int32)
        for i in range(rows + 1):
            for j in range(cols + 1):
                points[i * (cols + 1) + j][0] = j * s
                points[i * (cols + 1) + j][1] = i * s

        # Fill the rectangles
        for i in range(rows):
            for j in range(cols):
                col = self.get_random_color()
                cv.rectangle(board, (j * s, i * s), ((j + 1) * s, (i + 1) * s), col, -1)
        # cv.imshow("Checkerboard", board)

        # Warp the grid using an affine transformation
        # The parameters of the affine transformation are a bit constrained
        # to get transformations not too far-fetched
        scale = 0.5 + np.random.rand() * 1.5
        affine_transform = np.array([[scale * max(np.random.rand(), 0.7),
                                      scale * min(np.random.rand(), 0.3),
                                      np.random.randint(50)],
                                     [scale * min(np.random.rand(), 0.3),
                                      scale * max(np.random.rand(), 0.7),
                                      np.random.randint(50)]])
        warped_board = np.transpose(cv.warpAffine(board,
                                                  affine_transform,
                                                  img.shape[0:2]))
        points = np.transpose(np.concatenate((points,
                                              np.ones(((rows + 1) * (cols + 1), 1))),
                                             axis=1))
        warped_points = np.transpose(np.dot(affine_transform, points))
        warped_points = warped_points.astype(int)
        points = []  # keep only the points inside the image
        for i in range((rows + 1) * (cols + 1)):
            if warped_points[i][0] >= 0 and warped_points[i][0] < img.shape[0]\
               and warped_points[i][1] >= 0 and warped_points[i][1] < img.shape[1]:
                points.append([warped_points[i][1], warped_points[i][0]])
        points = np.array(points)
        # cv.imshow("Warped checkerboard", warped_board)

        # Add the warped checkerboard to img
        mask = np.stack([warped_board, warped_board, warped_board], axis=2)
        locs = np.where(mask != 0)
        img[locs[0], locs[1]] = mask[locs[0], locs[1]]
        return points

    def draw_cube(self, img):
        """ Draw a 2D projection of a cube and output the corners that are visible """
        # Generate a cube and apply to it an affine transformation
        # The order matters!
        # Two adjacent vertices differs only from one bit (as in Gray codes)
        min_dim = min(img.shape[:2])
        lx = 30 + np.random.rand() * min_dim / 3  # dimensions of the cube
        ly = 30 + np.random.rand() * min_dim / 3
        lz = 30 + np.random.rand() * min_dim / 3
        cube = np.array([[0, 0, 0],
                         [lx, 0, 0],
                         [0, ly, 0],
                         [lx, ly, 0],
                         [0, 0, lz],
                         [lx, 0, lz],
                         [0, ly, lz],
                         [lx, ly, lz]])
        rot_angles = np.random.rand(3) * 2 * math.pi
        rotation_1 = np.array([[math.cos(rot_angles[0]), -math.sin(rot_angles[0]), 0],
                               [math.sin(rot_angles[0]), math.cos(rot_angles[0]), 0],
                               [0, 0, 1]])
        rotation_2 = np.array([[1, 0, 0],
                               [0, math.cos(rot_angles[1]), -math.sin(rot_angles[1])],
                               [0, math.sin(rot_angles[1]), math.cos(rot_angles[1])]])
        rotation_3 = np.array([[math.cos(rot_angles[2]), 0, -math.sin(rot_angles[2])],
                               [0, 1, 0],
                               [math.sin(rot_angles[2]), 0, math.cos(rot_angles[2])]])
        scaling = np.array([[0.3 + np.random.rand() * 0.7, 0, 0],
                            [0, 0.3 + np.random.rand() * 0.7, 0],
                            [0, 0, 0.3 + np.random.rand() * 0.7]])
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

        # Find one point which is not on the boundary of the convex hull
        # (the hidden corner for example)
        cube = cube[:, :2]  # project on the plane z=0
        convex_hull = ConvexHull(cube)
        boundary = np.sort(convex_hull.vertices)
        bpoint = 0
        for i in range(8):
            if i != boundary[i]:  # i is not on the boundary
                bpoint = i
                break
        cube = cube.astype(int)
        # Compute point in the foreground (opposite from bpoint)
        fbitset = Bitset(bpoint, 3)
        for i in [0, 1, 2]:
            fbitset[i] = not fbitset[i]
        fpoint = int(fbitset)
        points = np.concatenate((cube[:bpoint, :], cube[(bpoint + 1):, :]), axis=0)

        # Get the three visible faces
        # faces contain the indices of the corners of the 3 faces starting from fpoint
        faces = np.zeros((3, 4), np.uint8) + fpoint
        for i in [0, 1, 2]:
            current_bit = Bitset(fpoint, 3)
            idx = 1
            for j in [0, 1, 0]:
                current_bit[(i + j) % 3] = not current_bit[(i + j) % 3]
                faces[i][idx] = int(current_bit)
                idx += 1
        # print(faces)

        # Fill the faces and draw the contours
        col_face = self.get_random_color()
        for i in [0, 1, 2]:
            cv.fillPoly(img, [cube[faces[i]].reshape((-1, 1, 2))],
                        (col_face, col_face, col_face))
        col_edge = (col_face + 128) % 256  # color that constrats with the face color
        thickness = np.random.randint(1, 3)
        for i in [0, 1, 2]:
            for j in [0, 1, 2, 3]:
                cv.line(img, (cube[faces[i][j], 0], cube[faces[i][j], 1]),
                        (cube[faces[i][(j + 1) % 4], 0], cube[faces[i][(j + 1) % 4], 1]),
                        (col_edge, col_edge, col_edge), thickness)

        # Keep only the points inside the image
        new_points = []
        for i in range(points.shape[0]):
            if points[i][0] >= 0 and points[i][0] < img.shape[1] and \
               points[i][1] >= 0 and points[i][1] < img.shape[0]:
                new_points.append([points[i][0], points[i][1]])
        points = np.array(new_points)
        return points

    def gaussian_noise(self, img):
        """ Apply Gaussian noise to the image """
        mean = 128
        var = 100
        cv.randn(img, mean, var)
        img[:, :, 1] = img[:, :, 0]
        img[:, :, 2] = img[:, :, 0]
        img = cv.blur(img, (2, 2))
        return np.array([])

    def draw_shape(self, img):
        """ Draw a shape randomly """
        idx = np.random.randint(7)
        if idx == 0:
            return self.draw_lines(img)
        if idx == 1:
            return self.draw_multiple_polygons(img)
        if idx == 2:
            return self.draw_ellipses(img)
        if idx == 3:
            return self.draw_star(img)
        if idx == 4:
            return self.draw_checkerboard(img)
        if idx == 5:
            return self.gaussian_noise(img)
        else:
            return self.draw_cube(img)

    def draw_interest_points(self, img, points):
        """ Draw in green the interest points """
        for i in range(points.shape[0]):
            cv.circle(img, (points[i][0], points[i][1]), 2, (0, 255, 0), 1)
