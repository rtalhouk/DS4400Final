import numpy as np
import pandas as pd
import os
import time
from PIL import Image
from keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import train_test_split
from typing import Tuple, Generator


class ImageLoader:
    """
    A class for loading the image dataset.
    """
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
        :param size: the size images should be resized to
        :param grayscale: whether or not to grayscale the images
        :param seed: a random seed
        :param train_size: The ratio of the training dataset
        :param shuffle: whether or not to shuffle and split the dataset into training and testing
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
        """
        Load images for this dataset.

        :return: self
        """
        if self.train_images is not None:
            return self

        # Allocate space for every image. This is done in one operation in order to avoid copying the entire
        # dataset every time a new image is added.
        print("Allocating image space...")
        size = self.get_dataset_size(self.image_dir, self.size)
        images = pd.DataFrame(data=np.zeros(size, dtype=np.float32),
                              columns=[i for i in range(self.size * self.size)])
        classes = pd.Series(data=np.zeros(size[0], dtype=np.byte), index=images.index)
        next_idx = (i for i in images.index)
        print("Done. Loading image sets.")

        # Load the images in each directory
        start = time.time()
        for entry in os.scandir(self.image_dir):
            self._load_subdir(entry.name, entry.path, images, classes, self.grayscale, self.size, next_idx)
            print(entry.name, "loaded.")
        print("Done loading images, took", (time.time() - start) / 60, "minutes")

        # Shuffle and split data if specified, otherwise, just set all data in training vars
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
    def get_dataset_size(image_dir: str, size: int) -> Tuple[int, int]:
        """
        Gets the size of the dataset

        :param image_dir: The directory containing image class directories
        :param size: The size of the width or height of an image
        :return: A tuple for the size of the dataframe required
        """
        img_size = size * size
        img_count = 0
        for entry in os.scandir(image_dir):
            img_count += len(os.listdir(entry.path))

        return img_count, img_size

    @staticmethod
    def _load_subdir(letter: str, letter_dir: str, images: pd.DataFrame, classes: pd.Series, grayscale: bool,
                     size: int, next_idx: Generator) -> None:
        """
        Load the letters in the given letter subdirectory.

        :param letter: The name of the letter
        :param letter_dir: The path to the directory to load images from
        :param images: The image dataframe
        :param classes: The classes series
        :param grayscale: Whether or not images should be grayscaled
        :param size: The size of each image
        :param next_idx: A generator of indexes to insert into next
        :return:
        """
        # Get the images in the directory
        for img_entry in os.scandir(letter_dir):
            # Open the image
            img = Image.open(img_entry.path, "r")
            width, height = img.size
            # Crop the image in the center if necessary
            if width != height:
                new_size = min(width, height)
                left = (width - new_size) / 2
                top = (height - new_size) / 2
                right = (width + new_size) / 2
                bottom = (width + new_size) / 2
                img = img.crop((left, top, right, bottom))
            # Resize if size is not 'size' size
            if min(width, height) != size:
                img = img.resize((size, size))
            # Grayscale the image
            if grayscale:
                img = img.convert("L")

            # Scale the image pixels
            img_data = np.array(img.getdata()) / 255.0

            # Insert the image and class into the dataframe and series
            curr_idx = next(next_idx)
            images.loc[curr_idx] = img_data
            if classes is not None:
                classes.loc[curr_idx] = letter

    @staticmethod
    def load_images_for_keras(image_dir: str = "../images/asl_alphabet_train/asl_alphabet_train",
                              color_mode: str = "grayscale", image_size: Tuple[int, int] = (80, 80),
                              validation_split: float = .2, seed: int = 1234321) -> Tuple[any, any]:
        """
        Get the Keras dataset for the images.

        :param image_dir: The directory of the images
        :param color_mode: Whether or not to grayscale
        :param image_size: The size images should be scaled to
        :param validation_split: How to split the training and validation sets by
        :param seed: Random seed
        :return: A tuple of the training and validation datasets
        """
        train = image_dataset_from_directory(image_dir, labels="inferred", color_mode=color_mode,
                                             image_size=image_size, validation_split=validation_split,
                                             subset="training", seed=seed)
        valid = image_dataset_from_directory(image_dir, labels="inferred", color_mode=color_mode,
                                             image_size=image_size, validation_split=validation_split,
                                             subset="validation", seed=seed)
        return train, valid
