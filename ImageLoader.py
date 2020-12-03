import numpy as np
import pandas as pd
import os
import time
from PIL import Image
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class ImageLoader:
    def __init__(self, image_dir: str = "./images/asl_alphabet_train/asl_alphabet_train/",
                 grayscale: bool = True, image_size: int = 80, num_class=False) -> None:
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
        self.grayscale = grayscale
        self.scaler = StandardScaler(copy=False)
        if num_class:
            self.class_map = {chr(letter): idx for idx, letter in enumerate(range(ord("A"), ord("Z") + 1))}
            self.class_map["del"] = len(self.class_map)
            self.class_map["space"] = len(self.class_map)
            self.class_map["nothing"] = len(self.class_map)
        else:
            self.class_map = {chr(letter): chr(letter) for letter in range(ord("A"), ord("Z") + 1)}
            self.class_map["del"] = "del"
            self.class_map["space"] = "space"
            self.class_map["nothing"] = "nothing"
            

    def load_images(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.images is not None:
            return self.images, self.classes

        print("Allocating image space...")
        size = self.get_dataset_size(self.image_dir, self.size)
        self.images = pd.DataFrame(data=np.zeros(size, dtype=np.float32), columns=[i for i in range(self.size * self.size)])
        self.classes = pd.Series(data=np.zeros(size[0], dtype=np.byte), index=self.images.index)
        next_idx = (i for i in self.images.index)
        print("Done. Loading image sets.")

        start = time.time()
        for entry in os.scandir(self.image_dir):
            self._load_subdir(entry.name, entry.path, self.images, self.classes, self.grayscale, self.size,
                              next_idx, self.class_map)
            print(entry.name, "loaded.")
        print("Done loading images, took", (time.time() - start) / 60, "minutes")
        self.scaler.fit(self.images)
        return self.images, self.classes

    def scale_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        start = time.time()
        if data is None:
            data = self.images
        self.scaler.transform(data, copy=False)
        print("Scaler finished in:", time.time() - start, "seconds")
        return data

    @staticmethod
    def get_dataset_size(image_dir, size) -> Tuple[int, int]:
        img_size = size * size
        img_count = 0
        for entry in os.scandir(image_dir):
            img_count += len(os.listdir(entry.path))

        return img_count, img_size

    @staticmethod
    def _load_subdir(letter, letter_dir, images, classes, grayscale, size, next_idx, class_map):
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
                classes.loc[curr_idx] = class_map[letter]
