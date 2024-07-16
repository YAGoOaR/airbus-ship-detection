import tensorflow as tf, numpy as np
from keras.saving import load_model
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

from my_utils.custom_loss import dice_coefficient, dice_coef_loss, IoU # Custom loss and metrics
from my_utils.data_preparation import load_image # data preprocessing

# Load configuration
config = pd.read_json('config.json', typ='series', dtype=str)
INPUT_FOLDER = config['input_path']
OUTPUT_FOLDER = config['output_path']

# Custom objects (that were used when compiling a model) are needed to be specified to restore the model.
custom_objects = {
    "IoU": IoU,
    "dice_coefficient": dice_coefficient,
    "dice_coef_loss": dice_coef_loss
}

# Loading the model using custom metrics
model: tf.keras.Model = load_model("./trained_models/384x384_1/model.keras", custom_objects=custom_objects)

image_names = os.listdir(INPUT_FOLDER)
image_paths = [os.path.join(INPUT_FOLDER, name) for name in image_names]

# Use same image height and width as in the model's input layer
image_size = model.input.shape[1:3]

if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

for image_path in image_paths:
    img = load_image(image_path, image_size)
    input_image_BGR = np.expand_dims(img, axis=0)
    img_rgb = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB) # Convert to RGB (the model uses BGR)

    output_mask: np.ndarray = model.predict(input_image_BGR, verbose=0) # Predict ships
    output_mask = (output_mask > 0.3).astype(int) # Set pixel confidence threshold to 0.3
    output_mask = np.squeeze(output_mask, axis=0)

    fig, axs = plt.subplots(2, figsize=(7, 12))

    # Display image and predicted ships
    axs[0].imshow(img_rgb)
    axs[0].axis("off")
    
    axs[1].imshow(output_mask)
    axs[1].axis('off')

    # Save to output
    plt.savefig(os.path.join(OUTPUT_FOLDER, os.path.basename(image_path)))
    plt.close(fig)
