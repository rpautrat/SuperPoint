import numpy as np
import tensorflow as tf
import cv2 as cv
import os
from pathlib import Path

from superpoint.models.utils import sample_homography, flat2mat
from superpoint.settings import DATA_PATH


seed = None


if __name__ == '__main__':
    tf.set_random_seed(seed)
    base_path = Path(DATA_PATH, 'COCO/val2014/')
    image_paths = list(base_path.iterdir())
    output_dir = Path(DATA_PATH, 'COCO/patches/')
    if not output_dir.exists():
        os.makedirs(output_dir)

    print("Generating patches of Coco val...")
    sess = tf.InteractiveSession()
    for num, path in enumerate(image_paths):
        new_path = Path(output_dir, str(num))
        if not new_path.exists():
            os.makedirs(new_path)

        # Read the image
        image = tf.read_file(str(path))
        image = tf.image.decode_png(image, channels=3)

        # Warp the image
        H = sample_homography(tf.shape(image)[:2])
        warped_image = tf.contrib.image.transform(image, H, interpolation="BILINEAR")
        H = flat2mat(H)[0, :, :]

        # Run
        im, warped_im, homography = sess.run([image, warped_image, H])

        # Write the result in files
        cv.imwrite(str(Path(new_path, "1.jpg")), im)
        cv.imwrite(str(Path(new_path, "2.jpg")), warped_im)
        np.savetxt(Path(new_path, "H_1_2"), homography, '%.5g')
    print("Files generated in " + str(output_dir))
