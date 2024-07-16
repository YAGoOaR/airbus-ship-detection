
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# To reduce code complexity, I split code into these files:
from src.custom_loss import dice_coefficient, dice_coef_loss, IoU
from src.plotting import save_history_plot
from src.data_preparation import DataGenerator, split_data
from src.my_model import build_model

# Here we define the path to our config, data and other variables
config = pd.read_json('config.json', typ='series', dtype=str)
DATASET: str = config['dataset_path']
TRAIN = f'{DATASET}/train_v2'
SEGMENTATION = f'{DATASET}/train_ship_segmentations_v2.csv'

batch_size = 4 # Batch size is set to 4 to fit my graphics card capabilities
image_size = (384, 384) # Here I use half size of the initial images to speed up the training

# Let's load the encoded segmentation data and corresponding image names
segmentation_df = pd.read_csv(SEGMENTATION)

# Split data and group masks
# To fasten training and balance the data, I decided to drop all images that have zero ships
train, val = split_data(segmentation_df, empty_image_ratio=0, test_size=0.1)

# Generators prepare images and decode masks (see my_utils/data_preparation.py)
train_generator = DataGenerator(
    image_folder=TRAIN,
    data=train,
    batch_size=batch_size,
    image_size=image_size
)

val_generator = DataGenerator(
    image_folder=TRAIN,
    data=val,
    batch_size=batch_size,
    image_size=image_size
)

# Let's output the amount of data we got:
print(f'''
Total segmented samples: {len(segmentation_df)}
Train batches: {len(train_generator)}
Train samples: {len(train_generator.data)}
Validation samples: {len(train_generator.data)}
''')

# Build a U-net model (see my_model.py for definition)
model = build_model(image_size=image_size)

# Here I use callbacks for controlling the learning rate and stopping training before overfitting
callbacks_list = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=1e-8),
    EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=4),
]

# Let's compile the model using default Adam optimizer and our custom metrics
# I use dice coefficient loss instead of the binary loss because it is very useful when data is disbalanced this much
# Also I increased learning rate (compared to the default 0.01) because we have
# ReduceLROnPlateau callback that will decrease it over time. Obviously, this way the training is faster
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.007), loss=dice_coef_loss, metrics=[dice_coefficient, IoU])

# 30 epochs is sufficient to finish the training
epochs = 30

# Training the U-net with epochs = 30, lr = 0.007
history = model.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=callbacks_list, shuffle=True)

# Save U-net to use it in inference.py
model.save('model.keras')

# Plot training and validation loss and dice coefficient
save_history_plot(history)
