from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_dataset(dataset_dir_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    print("Loading dataset")
    x, y = [], []
    for i, class_dir in enumerate(sorted(dataset_dir_path.iterdir())):
        for file in class_dir.iterdir():
            img_file = cv2.imread(str(file))  # TODO: Check if reading as GRAYSCALE image isn't better
            x.append(img_file)
            y.append(i)

    return np.asarray(x), np.asarray(y)


def show_images_and_labels(images, labels):
    for image, label in zip(images, labels):
        cv2.imshow("image", image)
        print(label)
        cv2.waitKey()
