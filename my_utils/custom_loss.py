import keras.backend as K

def IoU(y_true, y_pred, epsilon=1e-6):
    I = K.sum(y_true * y_pred, axis=(1, 2))
    U = K.sum(y_true + y_pred, axis=(1, 2)) - I
    return (I + epsilon) / (U + epsilon)

def dice_coefficient(y_true, y_pred, epsilon=1):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    I = K.sum(y_true * y_pred)
    return (2. * I + epsilon) / (K.sum(y_true) + K.sum(y_pred) + epsilon)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)
