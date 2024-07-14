import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

from my_utils.custom_loss import dice_coefficient, dice_coef_loss, IoU
from my_utils.data_preparation import load_image
from keras.saving import load_model

import random

DATASET = './airbus-ship-detection'

custom_objects = {
    "IoU": IoU,
    "dice_coefficient": dice_coefficient,
    "dice_coef_loss": dice_coef_loss
}

model: tf.keras.Model = load_model("model.keras", custom_objects=custom_objects)

image_size = model.input.shape[1:3]

TEST = f'{DATASET}/test_v2'

def load_input(image_path):
    img = load_image(image_path, image_size)
    img = np.expand_dims(img, axis=0)
    return img

test_images_with_ships = [8, 10, 40, 43, 44, 46, 47, 50, 60, 64, 66, 67, 68, 74, 75, 76, 98]

columns = 2
rows = 6
fig, axs = plt.subplots(rows, columns, figsize=(7, 12))

selected_images = random.shuffle(test_images_with_ships)[:rows]

for i, img_idx in enumerate(selected_images):
    img_path = os.path.join(TEST, os.listdir(TEST)[img_idx])
    output_mask: np.ndarray = model.predict(load_input(img_path), verbose=0)
    output_mask = (output_mask > 0.3).astype(int)

    img = load_image(img_path, image_size)
    img_rgb = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)

    mask = np.squeeze(output_mask, axis=0)

    axs[i, 0].imshow(img_rgb)
    axs[i, 0].axis("off")
    
    axs[i, 1].imshow(mask)
    axs[i, 1].axis('off')

fig.subplots_adjust(wspace = 0, hspace=0.01)
plt.show()
