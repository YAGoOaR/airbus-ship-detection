from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Conv2DTranspose, Concatenate, ReLU, Activation, Input
from keras.models import Model
from keras.activations import sigmoid

# Here is a U-net based model
# However, I decided to make it's size smaller to fit our needs for the solution

# Just simple 3x3 convolution. 
# In the original U-net architecture batch normalization is not used.
# However, I decided to add batch normalization to see probable improvements.
def convolution_3x3(x, filters: int):
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

# A block that consists of 2 identical convolutions, with same filter count
def convolution_block(x, filters: int):
    x = convolution_3x3(x, filters)
    x = convolution_3x3(x, filters)
    return x

# U-net downsampling block (also returns skip features)
def encoding_block(x, filters: int):
    x = convolution_block(x, filters)
    skip_features = x
    x = MaxPool2D((2, 2))(x)
    return skip_features, x

# U-net upsampling block
def decoding_block(x, skip_features, filters: int):
    x = Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)
    x = Concatenate()([x, skip_features])
    x = convolution_block(x, filters)
    return x

# Build complete U-net
def build_model(image_size: tuple[int, int] = (512, 512)) -> Model:
    input = Input((*image_size, 3))
    
    x = input
    # Encoder - model contractive path
    skip_1, x = encoding_block(x, 32)
    skip_2, x = encoding_block(x, 64)
    skip_3, x = encoding_block(x, 128)
    skip_4, x = encoding_block(x, 256)
    x = convolution_block(x, 512) # Increasing feature information but without maxpooling

    # Decoder - model's expansive pathway
    x = decoding_block(x, skip_4, 256)
    x = decoding_block(x, skip_3, 128)
    x = decoding_block(x, skip_2, 64)
    x = decoding_block(x, skip_1, 32)

    x = Conv2D(1, 1, padding='same')(x) # Reduce dimensionality to a single feature map
    output = Activation(sigmoid)(x) # Use sigmoid activation because in this task we classify pixels

    return Model(input, output)

if __name__ == "__main__":
    print(build_model().summary())
