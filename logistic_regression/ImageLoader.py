import numpy as np
import pandas as pd
import os
import time
import random
from PIL import Image
from keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import train_test_split
from typing import Tuple


class ImageLoader:
    def __init__(self, image_dir: str = "../images/asl_alphabet_train/asl_alphabet_train/",
                 grayscale: bool = True, image_size: int = 80,
                 seed: int = 12345, train_size: float = .8, shuffle=True) -> None:
        """
        Prepares the image loader with the directory of images.

        The format of the directory MUST be:
        - <root>/
            - A/
            - B/
            - .../
            - nothing/
            - space/
            - del/

        :param image_dir: The directory storing the images
        """
        self.image_dir = image_dir
        self.size = image_size
        self.train_images = None
        self.train_classes = None
        self.grayscale = grayscale
        self.seed = seed
        self.test_images = None
        self.test_classes = None
        self.train_size = train_size
        self.shuffle = shuffle

    def load_images(self) -> "ImageLoader":
        if self.train_images is not None:
            return self

        print("Allocating image space...")
        size = self.get_dataset_size(self.image_dir, self.size)
        images = pd.DataFrame(data=np.zeros(size, dtype=np.float32),
                              columns=[i for i in range(self.size * self.size)])
        classes = pd.Series(data=np.zeros(size[0], dtype=np.byte), index=images.index)
        next_idx = (i for i in images.index)
        print("Done. Loading image sets.")

        start = time.time()
        for entry in os.scandir(self.image_dir):
            self._load_subdir(entry.name, entry.path, images, classes, self.grayscale, self.size, next_idx)
            print(entry.name, "loaded.")
        print("Done loading images, took", (time.time() - start) / 60, "minutes")

        if self.shuffle:
            train_x, test_x, train_y, test_y = train_test_split(images, classes, random_state=self.seed,
                                                                train_size=self.train_size)
            self.train_images = train_x
            self.train_classes = train_y
            self.test_images = test_x
            self.test_classes = test_y
        else:
            self.train_images = images
            self.train_classes = classes
        return self

    @staticmethod
    def get_dataset_size(image_dir, size) -> Tuple[int, int]:
        img_size = size * size
        img_count = 0
        for entry in os.scandir(image_dir):
            img_count += len(os.listdir(entry.path))

        return img_count, img_size

    @staticmethod
    def _load_subdir(letter, letter_dir, images, classes, grayscale, size, next_idx):
        for img_entry in os.scandir(letter_dir):
            img = Image.open(img_entry.path, "r")
            width, height = img.size
            if width != height:
                new_size = min(width, height)
                left = (width - new_size) / 2
                top = (height - new_size) / 2
                right = (width + new_size) / 2
                bottom = (width + new_size) / 2
                img = img.crop((left, top, right, bottom))
            if min(width, height) != size:
                img = img.resize((size, size))
            if grayscale:
                img = img.convert("L")

            img_data = np.array(img.getdata()) / 255.0
            curr_idx = next(next_idx)
            images.loc[curr_idx] = img_data
            if classes is not None:
                classes.loc[curr_idx] = letter

    @staticmethod
    def load_images_for_keras(image_dir: str = "../images/asl_alphabet_train/asl_alphabet_train",
                              color_mode: str = "grayscale", image_size: Tuple[int, int] = (80, 80),
                              validation_split: float = .2, seed: int = 1234321):
        train = image_dataset_from_directory(image_dir, labels="inferred", color_mode=color_mode,
                                             image_size=image_size, validation_split=validation_split,
                                             subset="training", seed=seed)
        valid = image_dataset_from_directory(image_dir, labels="inferred", color_mode=color_mode,
                                             image_size=image_size, validation_split=validation_split,
                                             subset="validation", seed=seed)
        return train, valid
