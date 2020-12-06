import numpy as np
import pandas as pd
import os
import time
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class ImageLoader:
    def __init__(self, image_dir: str = "./images/asl_alphabet_train/asl_alphabet_train/",
                 grayscale: bool = True, image_size: int = 80) -> None:
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
        self.images = None
        self.classes = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.grayscale = grayscale
        self.scaler = StandardScaler(copy=True)

    def load_images(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.x_train is not None:
            return self.x_train, self.x_test, self.y_train, self.y_test

        print("Allocating image space...")
        size = self.get_dataset_size(self.image_dir, self.size)
        self.images = pd.DataFrame(data=np.zeros(size), columns=[i for i in range(self.size * self.size)])
        self.classes = pd.Series(data=np.full(size[0], "None"), index=self.images.index)
        next_idx = (i for i in self.images.index)
        print("Done. Loading image sets.")

        start = time.time()
        for entry in os.scandir(self.image_dir):
            self._load_subdir(entry.name, entry.path, self.images, self.classes, self.grayscale, self.size,
                              next_idx)
            print(entry.name, "loaded.")
        print("Done loading images, took", (time.time() - start) / 60, "minutes")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.images, self.classes, test_size=0.2)
        self.scaler.fit(self.x_train)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def scale_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        start = time.time()
        if data is None:
            res = self.scaler.transform(self.images)
        else:
            res = self.scaler.transform(data)
        print("Scaler finished in:", time.time() - start, "seconds")
        return res

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

            img_data = list(img.getdata())
            curr_idx = next(next_idx)
            images.loc[curr_idx] = img_data
            if classes is not None:
                classes.loc[curr_idx] = letter
