# Live Classifier: Real-Time Object Classification

## Project Overview

This project implements a real-time object classification system using a Convolutional Neural Network (CNN). The system classifies objects in a live video feed into predefined classes: remote control, cell phone, TV, and coffee mug. The classifier processes video frames in real-time, displaying the predicted class and confidence level for detected objects.

## Setup Instructions

1. Install required libraries:
- pip install opencv-python tensorflow numpy pillow
2. Download the pre-trained model:
- Download `live_classifier_model.h5` from https://northeastern-my.sharepoint.com/:u:/r/personal/ambulkar_m_northeastern_edu/Documents/live_classifier_model.h5?csf=1&web=1&e=vLOsjF
- Place it in the `mini-project7` directory

## Usage Guide

1. Run the live classifier:
- python LiveClassifier.py -m live_classifier_model.h5

2. Optional arguments:
- `-f [video_file]`: Use a video file instead of webcam
- `-o [output_file]`: Specify an output video file (default: output.avi)

3. During execution:
- The application will display the live video feed with object classifications
- Press 'q' to quit the application

## Project Components

### 1. Dataset Collection and Preprocessing

- Dataset: 100 images per class (remote control, cell phone, TV, coffee mug)
- Preprocessing: Resizing to 224x224 pixels, normalization

### 2. CNN Model Architecture

Our CNN model for live object classification consists of the following layers:

1. Input Layer:
   - Shape: (224, 224, 3) - Accepting 224x224 RGB images

2. Convolutional Layers:
   - Conv2D: 32 filters, 3x3 kernel, ReLU activation
   - MaxPooling2D: 2x2 pool size
   - Conv2D: 64 filters, 3x3 kernel, ReLU activation
   - MaxPooling2D: 2x2 pool size
   - Conv2D: 64 filters, 3x3 kernel, ReLU activation
   - MaxPooling2D: 2x2 pool size

3. Flatten Layer:
   - Flattens the 3D output to 1D for dense layers

4. Dense Layers:
   - Dense: 64 units, ReLU activation
   - Dropout: 0.5 rate for regularization

5. Output Layer:
   - Dense: 4 units (matching number of classes), Softmax activation

Key Features:
- The model uses three convolutional layers with increasing filter sizes to extract hierarchical features from the input images.
- Max pooling layers help in reducing spatial dimensions and computational load.
- ReLU activation is used throughout for non-linearity, except in the output layer.
- Dropout is applied before the final layer to prevent overfitting.
- Softmax activation in the output layer provides probability distribution over the classes.

### 3. Model Training and Evaluation

- Training process: Training Accuracy = 97.45% with 50 epochs of default batch size 32
- Validation accuracy = 63.75%

### 4. Real-time Classification Implementation

The `LiveClassifier.py` script performs the following steps:
1. Captures video frames from webcam or video file
2. Preprocesses each frame (resize, normalize)
3. Applies the CNN model for classification
4. Displays the predicted class and confidence level
5. Shows FPS (Frames Per Second) for performance monitoring

## Files in the Repository

- `LiveClassifier.py`: Main script for real-time classification
- `preprocess_data.py`: Script for dataset preprocessing
- `train_model.py`: Script for training the CNN model
- `live_classifier_model.h5`: Trained model file

## Model and Dataset Access

- Dataset: https://northeastern-my.sharepoint.com/:f:/r/personal/ambulkar_m_northeastern_edu/Documents/CS5330-mini-project7_data?csf=1&web=1&e=Wp2KDM
- Trained Model: https://northeastern-my.sharepoint.com/:u:/r/personal/ambulkar_m_northeastern_edu/Documents/live_classifier_model.h5?csf=1&web=1&e=vLOsjF

## Video Demonstration

https://youtube.com/shorts/_oR95TbGVP4?feature=share
