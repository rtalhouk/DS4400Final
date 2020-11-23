import numpy as np
import pandas as pd
import os
import time
from PIL import Image
from typing import Tuple


class ImageLoader:
    def __init__(self, image_dir: str = "./images/asl_alphabet_train/asl_alphabet_train/",
                 grayscale: bool = True) -> None:
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
        self.size = 80
        self.images = None
        self.classes = None
        self.grayscale = grayscale
        self.next = None

    def load_images(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.images is not None:
            return self.images.iloc[:, :-1], self.images.iloc[:, -1]

        size = self.get_dataset_size()
        self.images = pd.DataFrame(data=np.zeros(size), columns=[i for i in range(self.size * self.size)])
        self.classes = pd.Series(data=np.full(size[0], "None"), index=self.images.index)
        self.next = (i for i in self.images.index)

        start = time.time()
        for entry in os.scandir(self.image_dir):
            self._load_subdir(entry.name, entry.path)
            print(entry.name, "loaded.")
        print("Done loading images, took", (time.time() - start) / 60, "minutes")
        return self.images, self.classes

    def get_dataset_size(self) -> Tuple[int, int]:
        img_size = self.size * self.size
        img_count = 0
        for entry in os.scandir(self.image_dir):
            img_count += len(os.listdir(entry.path))

        return img_count, img_size

    def _load_subdir(self, letter, letter_dir):
        i = -1
        for img_entry in os.scandir(letter_dir):
            i += 1
            if i % 4 != 0:
                continue
            img = Image.open(img_entry.path, "r")
            width, height = img.size
            if width != height:
                new_size = min(width, height)
                left = (width - new_size) / 2
                top = (height - new_size) / 2
                right = (width + new_size) / 2
                bottom = (width + new_size) / 2
                img = img.crop((left, top, right, bottom))
            if min(width, height) != self.size:
                img = img.resize((self.size, self.size))
            if self.grayscale:
                img = img.convert("L")

            img_data = list(img.getdata())
            curr_idx = next(self.next)
            self.images.loc[curr_idx] = img_data
            self.classes.loc[curr_idx] = letter
