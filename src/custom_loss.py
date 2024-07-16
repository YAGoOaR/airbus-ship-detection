import keras.backend as K

# Epsilon is needed for the function to work properly when A and B masks are empty
# Also epsilon can be raised to smooth out the function for small masks
def IoU(y_true, y_pred, epsilon=1e-6):
    '''
    Intersection over union: |A ∩ B| / |A ∪ B|, where A - actual, B - predicted.
    '''
    I = K.sum(y_true * y_pred, axis=(1, 2))
    U = K.sum(y_true + y_pred, axis=(1, 2)) - I
    return (I + epsilon) / (U + epsilon)

def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    '''
    Dice coefficient: 2|A ∩ B| / (|A| + |B|), where A - actual, B - predicted.
    '''

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    I = K.sum(y_true * y_pred)
    return (2. * I + epsilon) / (K.sum(y_true) + K.sum(y_pred) + epsilon)

# Here i just invert the Dice coefficient knowing it's range is [0; 1]
def dice_coef_loss(y_true, y_pred):
    '''
    Dice coefficient loss: 1 - (2|A ∩ B| / (|A| + |B|)), where A - actual, B - predicted.
    '''
    return 1 - dice_coefficient(y_true, y_pred)
