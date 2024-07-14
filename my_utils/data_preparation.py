
import os
import cv2
from keras.utils import Sequence
import pandas as pd
import numpy as np
from my_utils.encoding import decode_RLE

import albumentations as A # Augmentation
from sklearn.model_selection import train_test_split # Splitting to train and validation

# Let's use simple augmentation - only flipping vertically and horizontally
# I decided to not to use complex augmentations, because the data is already various enough
# We can use brightness/gamma/hue and geometric augmentations to further improve the score 
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])

# Get normalized and resized BGR image
def load_image(path: str, size: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = img / 255.0
    return img

# We need a generator to load and send data by batches.
# Obviously, we can't load such dataset entirely in our GPU memory.
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

    # Number of batches
    def __len__(self):
        return int(len(self.data) / self.batch_size)
    
    # Join data folder with image name
    def get_img_path(self, name: str) -> str:
        return os.path.join(self.image_folder, name)

    # Merges different ship masks into one mask (for each image)
    @staticmethod
    def merge_masks(masks: np.ndarray, imsize: tuple[int, int]) -> np.ndarray[float]:
        if len(masks) == 0:
            return np.zeros((imsize), dtype=float)
        else:
            return np.sum(masks, dtype=float, axis=0)

    # Get a single batch
    def __getitem__(self, index: int):
        batch_data = self.data[index * self.batch_size:(index + 1) * self.batch_size] # get data by index offset

        # Process input images (X) and corresponding masks (Y)
        X = (self.get_img_path(name) for name in batch_data.index) # get paths
        X = map(lambda path: load_image(path, self.image_size), X) # load images
        X = np.array([*X]) # Compute list, put into array

        Y = ([decode_RLE(mask, out_shape=self.image_size) for mask in masks] for masks in batch_data) # Decode each mask
        Y = (self.merge_masks(masks) for masks in Y) # Merge masks from same images
        Y = np.array([*Y])

        transformed = transform(image=X, mask=Y) # Augmentation transformation

        return transformed['image'], transformed['mask']

# Group ship masks by images
def group_masks(df: pd.DataFrame) -> pd.DataFrame:

    def masks_to_list(masks):
        lst = list(masks)
        # Empty segmented images have NaN values, so they should be filtered out:
        return lst if lst != [np.nan] else []

    return df.groupby('ImageId')['EncodedPixels'].agg(masks_to_list)

# Transform mask data to use for training and form training and validation sets
def split_data(data: pd.DataFrame, empty_image_ratio:float=None, test_size:float=0.2, random_state:int=42) -> tuple[pd.DataFrame, pd.DataFrame]:
    masks = group_masks(data) # Group ship masks by images

    # Separate data with and without ships
    mask_has_ships = masks.map(lambda x: len(x) > 0)
    with_ships = masks[mask_has_ships]
    without_ships = masks[~mask_has_ships]
    
    if empty_image_ratio != None:
        total_count = masks.shape[0]
        without_ships = without_ships.iloc[:int(total_count * empty_image_ratio)]
    
    # Combine the data
    masks = pd.concat([with_ships, without_ships], axis=0)

    # Get ship count to stratify the data
    ship_counts = masks.map(len)

    train, test = train_test_split(
        masks,
        test_size=test_size,
        stratify=ship_counts.array,
        random_state=random_state
    )

    return train, test
