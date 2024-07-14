
import os
import cv2
from keras.utils import Sequence
import pandas as pd
import numpy as np
from encoding import decode_RLE

import albumentations as A
from sklearn.model_selection import train_test_split

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])

def load_image(path: str, size: tuple[int, int]):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = img / 255.0
    return img

class DataGenerator(Sequence):
    def __init__(
            self, 
            image_folder: str, 
            data: pd.DataFrame, 
            batch_size: int = 4, 
            image_size: tuple[int, int]=(768, 768)
        ):

        self.image_folder = image_folder
        self.batch_size = batch_size
        self.image_size = image_size
        self.data = data

    def __len__(self):
        return int(len(self.data) / self.batch_size)

    def get_img_path(self, name: str):
        return os.path.join(self.image_folder, name)

    @staticmethod
    def merge_masks(masks: np.ndarray, imsize: tuple[int, int]) -> np.ndarray[float]:
        if len(masks) == 0:
            return np.zeros((imsize), dtype=float)
        else:
            return np.sum([cv2.resize(m, imsize) for m in masks], dtype=float, axis=0)

    def __getitem__(self, index: int):
        batch_data = self.data[index * self.batch_size:(index + 1) * self.batch_size]

        X = (self.get_img_path(name) for name in batch_data.index)
        X = np.array([*map(lambda path: load_image(path, self.image_size), X)])

        Y = ([decode_RLE(m) for m in masks] for masks in batch_data)
        Y = np.array([self.merge_masks(masks, self.image_size) for masks in Y])        

        transformed = transform(image=X, mask=Y)

        return transformed['image'], transformed['mask']

def group_masks(df: pd.DataFrame) -> pd.DataFrame:

    def masks_to_list(masks):
        lst = list(masks)
        return lst if lst != [np.nan] else []

    return df.groupby('ImageId')['EncodedPixels'].agg(masks_to_list)

def split_data(data: pd.DataFrame, empty_image_ratio:float=None, test_size:float=0.2, random_state:int=42) -> tuple[pd.DataFrame, pd.DataFrame]:
    masks = group_masks(data)

    mask_has_ships = masks.map(lambda x: len(x) > 0)
    with_ships = masks[mask_has_ships]
    without_ships = masks[~mask_has_ships]
    
    if empty_image_ratio != None:
        total_count = masks.shape[0]
        without_ships = without_ships.iloc[:int(total_count * empty_image_ratio)]
    
    masks = pd.concat([with_ships, without_ships], axis=0)

    ship_counts = masks.map(lambda mask_list: len(mask_list))

    train, test = train_test_split(
        masks,
        test_size=test_size,
        stratify=ship_counts.array,
        random_state=random_state
    )

    return train, test
