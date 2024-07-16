# Airbus ship detection task

In this solution, I have built a U-Net model that performs image segmentation.

### Chosen tools
The tools recommended for the task were python, Keras (with tensorflow backend), and U-net model with Dice score metric.
Along with these tools, I decided to use pandas for data loading and processing data frames, OpenCV for loading and processing images, scikit learn to stratify and split data, matplotlib to draw images and graphs. 

You can use the [requirements.txt](requirements.txt) to recover the complete conda environment with all installed libraries. However, there is a more concise list (that only includes libraries that are used in code), that was made using `pipreqs`:
```pip
albumentations==1.4.1
keras==2.13.1
matplotlib==3.8.4
numpy==1.26.4
opencv-python==4.10.0.84
opencv-python_headless==4.10.0.84
pandas==2.2.2
scikit-learn==1.4.2
scikit-image==0.23.2
tensorflow==2.13.0
```
Python version: 3.10.13

### EDA
> [View notebook](EDA/EDA.ipynb)

In the exploratory data analysis I discovered features of the given data.
Input images have 768x768 size, and RGB channels. Decoded segmentation masks are 768x768, binary color.

Examples from data:

![Examples](https://github.com/YAGoOaR/Images/blob/main/PythonSegmentation/examples.png?raw=true)

Mask images are encoded into Run-Length format. As the name implies, such format consists of runs and lengths that paint the pixels of an image.
The data has big imbalance. There are much more images where the ships are absent than images where there are ships. The pixel class imbalance is even greater.
To reduce training time and balance the dataset better, I dropped images without annotated ships.

### Preparing the data
> [View data preparation](my_utils/data_preparation.py)

At first, I read images using opencv library. Since it returns data in BGR format, I decided to use the same channel order in the model.
Segmentation data needs to be decoded first, then different ship masks should be merged into one for each image. This way the model sees all ships at once and can correctly compute loss and train.
It is always beneficial to augment the training data. In this project I decided to only use horizontal and vertical image flipping because I found it sufficient to get good results. We can easily add brightness/gamma/hue/spatial augmentations to make the score even better, but the training should be done once again.

### Creating a model
> [View model code](my_model.py)

The model architecture used in this project is U-net. I slightly changed it - reduced the model and input size to improve training time.
The input image size is 384x384 now (half the original size). It can impact only the smallest ships. However most of the data has clearly visible ships.

<!-- TODO: add an illustration -->
<!-- ![model_visualization](pictures/model.png) -->

The model is basically an autoencoder but with skip features.
As we know, autoencoder consists of an encoder and a decoder.
Encoder increases feature information, reducing the image dimensionality. 
The decoder combines the features and spatial information (using skip features from the encoder), upsampling the segmentation data.

### Training the model
> [View training code](training.py)

The model was trained using ReduceLrOnPlateau and EarlyStopping keras callbacks.
ReduceLrOnPlateau helps to control the learning rate when it is needed (when model's paremeters reach local loss function plateau).
Early stopping (based on validation score) makes it easier not to overfit the model by stopping the training in time. 

For the model optimization, I used Adam optimizer with learning rate 0.007 (further decreased by ReduceLrOnPlateau).
Batch size is 4 to fit my GPU capabilities.
Training process took 40 epochs, 10 hours total.

![training_history](https://github.com/YAGoOaR/Images/blob/main/PythonSegmentation/training_history.png?raw=true)

During the training process, the model reached Dice score of 0.867. 
For further improvements, I would use more augmentation methods in the pipeline (e.g. random hue/brightness/gamma/noise, spatial distortions, random cropping & resizing, etc.), and a pretrained encoder for the U-net model, such as VGG, ResNet, etc.

### Results
The model succesfully predicts ships on images:

![result_demo](https://github.com/YAGoOaR/Images/blob/main/PythonSegmentation/result_demo.png?raw=true)

> View output demo to see more prediction results.
> [Demo outputs](output_demo)

The model may sometimes struggle with ships that have a really tiny size that I can barely see with my eyes. That's because I reduced the input size to half the initial size so such ship does not cover much pixels in the model's input. The solution can be very simple. 768x768 image can be divided into 4 384x384 images and processed separately. Or just zoom in better when making such photos. The problem is that I can't train a model on bigger image size while not having sufficient computational resources. That's why I decided to train on 384x384.
Also the model may have some problems when the ship is moored to a prier. The model may notice ship in such cases but have some problems segmenting it correctly, separating from all other objects and garbage near the prier/shore.
But overall, I am very satisfied by the result of a my model, taking into account the resources I had for training it.

### To improve results further:
- Increase model/data size and increase training time
- Include empty dataset images too - it will help the model to reduce false positive preditions (though true negatives are much more common because of the imbalance)
- U-net can be merged with a pretrained encoder (e.g. ResNet)
- Use more augmentation methods (e.g. random hue/brightness/gamma/noise, spatial distortions, cropping & resizing to have more various zoom, etc.)

### Usage
1. Load the same conda environment using the `requirements.txt`.
2. Set `input_path` field in `config.json` to define the input data directory. Also set the `output_path`.
3. Set `inference_draw_input_image_too` to true if you want to draw both input and output images.
4. Run `inference.py` to get predictions.

> [See inference code](inference.py)
