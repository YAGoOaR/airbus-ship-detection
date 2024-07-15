import tensorflow as tf, numpy as np
from keras.saving import load_model
import cv2
import matplotlib.pyplot as plt
import os, random

from my_utils.custom_loss import dice_coefficient, dice_coef_loss, IoU # Custom loss and metrics
from my_utils.data_preparation import load_image # data preprocessing

# Path to data
config_path = 'dataset_path.txt'
with open(config_path, 'r') as file:
    DATASET = file.read().strip()
    
TEST = f'{DATASET}/test_v2'

# We are not interested in empty images with water only
# That's why i found and defined the images in the test dataset (first 100 samples) that have visible ships:
test_images_with_ships = [8, 10, 40, 43, 44, 46, 47, 50, 60, 64, 66, 67, 68, 74, 75, 76, 98]

# Custom objects (that were used when compiling a model) are needed to be specified to restore the model.
custom_objects = {
    "IoU": IoU,
    "dice_coefficient": dice_coefficient,
    "dice_coef_loss": dice_coef_loss
}

# Loading the model using custom metrics
model: tf.keras.Model = load_model("./trained_models/384x384_1/model.keras", custom_objects=custom_objects)

# Use same image height and width as in the model's input layer
image_size = model.input.shape[1:3]

# Let's show 6 random test images
images_to_show = 6

# Select random images with ships
selected_indices = test_images_with_ships.copy()
random.shuffle(selected_indices)
selected_indices = selected_indices[:images_to_show]

# Get their paths
images = [os.path.join(TEST, os.listdir(TEST)[idx]) for idx in selected_indices]


def model_demo(images: list[str], model: tf.keras.Model, count: int = 6) -> None:
    '''
    Display images in 2 columns.
    Left column - input images.
    Right column - predicted masks.
    '''
    columns = 2
    fig, axs = plt.subplots(count, columns, figsize=(7, 12))

    for i, img_path in enumerate(images[:count]):
        # Prepare image
        img = load_image(img_path, image_size) 
        img_rgb = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB) # Convert to RGB (the model uses BGR)
        input_image = np.expand_dims(img, axis=0) # Add batch dimension

        output_mask: np.ndarray = model.predict(input_image, verbose=0) # Predict ships
        output_mask = (output_mask > 0.3).astype(int) # Set pixel confidence threshold to 0.3

        mask = np.squeeze(output_mask, axis=0) # Remove batch dimension

        # Display image and predicted ships
        axs[i, 0].imshow(img_rgb)
        axs[i, 0].axis("off")
        
        axs[i, 1].imshow(mask)
        axs[i, 1].axis('off')

    fig.subplots_adjust(hspace=0.01) # Make gaps between images smaller
    plt.show()

# Launch the demo
model_demo(images, model=model, count = images_to_show)
