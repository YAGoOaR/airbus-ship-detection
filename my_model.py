
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Conv2DTranspose, Concatenate, ReLU, Activation, Input
from keras.models import Model
from keras.activations import sigmoid

def conv_BN_block(x, filters):
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

def encoding_block(x, filters):
    x = conv_BN_block(x, filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoding_block(x, skip_layer, filters):
    x = Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)
    x = Concatenate()([x, skip_layer])
    x = conv_BN_block(x, filters)
    return x

def build_model(image_size: tuple[int, int] = (512, 512)) -> Model:
    input = Input((*image_size, 3))
    
    x = input
    skip_1, x = encoding_block(x, 32)
    skip_2, x = encoding_block(x, 64)
    skip_3, x = encoding_block(x, 128)
    skip_4, x = encoding_block(x, 256)

    x = conv_BN_block(x, 512)

    x = decoding_block(x, skip_4, 256)
    x = decoding_block(x, skip_3, 128)
    x = decoding_block(x, skip_2, 64)
    x = decoding_block(x, skip_1, 32)

    x = Conv2D(1, 1, padding='same')(x)
    output = Activation(sigmoid)(x)

    return Model(input, output)

if __name__ == "__main__":
    print(build_model().summary())
