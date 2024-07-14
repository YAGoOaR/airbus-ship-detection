import pandas as pd
import tensorflow as tf

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from myutils.loss import dice_coefficient, dice_coef_loss, IoU
from myutils.plotting import save_history_plot

from data_preparation import DataGenerator, split_data
from my_model import build_model

DATASET = './airbus-ship-detection'
TRAIN = f'{DATASET}/train_v2'
SEGMENTATION = f'{DATASET}/train_ship_segmentations_v2.csv'

batch_size = 4
image_size = (384, 384)

segmentation_df = pd.read_csv(SEGMENTATION)

train, val = split_data(segmentation_df, empty_image_ratio=0, test_size=0.1)

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

print(f'''
Total segmented samples: {len(segmentation_df)}
Train batches: {len(train_generator)}
Train samples: {len(train_generator.data)}
Validation samples: {len(train_generator.data)}
''')

model = build_model(image_size=image_size)

callbacks_list = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=1e-8),
    EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=4),
]

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.007), loss=dice_coef_loss, metrics=[dice_coefficient, IoU])

epochs = 30

history = model.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=callbacks_list, shuffle=True)

model.save('model.keras')

save_history_plot(history)
