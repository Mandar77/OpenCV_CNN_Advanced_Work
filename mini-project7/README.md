# Mini-Project 7: Live Classifier - Real-Time Object Classification

## Project Overview

This project implements a real-time object classification system using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The system classifies objects from a live video feed (or a video file) into four predefined classes: 'Remote Control', 'Cell Phone', 'TV', and 'Coffee Mug'. The classifier processes video frames in real-time, displaying the predicted class and confidence level on the video stream.

## Workflow

The project is divided into three main stages:

1.  **Data Preprocessing**: The `preprocess_data.py` script is used to load, resize, and normalize the image dataset. The processed data is saved as NumPy arrays.
2.  **Model Training**: The `train_model.py` script builds and trains the CNN model on the preprocessed data and saves the trained model as an HDF5 file (`.h5`).
3.  **Live Classification**: The `LiveClassifier.py` script loads the trained model and performs real-time classification on a live webcam feed or a video file.

## Setup Instructions

1.  **Install required libraries**:
    ```bash
    pip install opencv-python tensorflow numpy pillow
    ```
2.  **Download the dataset and pre-trained model**:
    -   **Dataset**: Download the image dataset from [this link](https://northeastern-my.sharepoint.com/:f:/r/personal/ambulkar_m_northeastern_edu/Documents/CS5330-mini-project7_data?csf=1&web=1&e=Wp2KDM) and place it in a `data` directory inside `mini-project7`.
    -   **Pre-trained Model**: If you don't want to train the model yourself, download the `live_classifier_model.h5` file from [this link](https://northeastern-my.sharepoint.com/:u:/r/personal/ambulkar_m_northeastern_edu/Documents/live_classifier_model.h5?csf=1&web=1&e=vLOsjF) and place it in the `mini-project7` directory.

## How to Run

### 1. Preprocess the Data (Optional)
If you want to train the model on your own data, first run the preprocessing script.
```bash
python preprocess_data.py
```
This will create `preprocessed_X.npy` and `preprocessed_y.npy`.

### 2. Train the Model (Optional)
Next, run the training script to create the model.
```bash
python train_model.py
```
This will create `live_classifier_model.h5`.

### 3. Run the Live Classifier
To run the real-time classifier, use the following command:
```bash
python LiveClassifier.py -m live_classifier_model.h5
```

**Optional arguments**:
-   `-f <video_file>`: Use a video file as input instead of the webcam.
-   `-o <output_file.avi>`: Specify a name for the output video file (default is `output.avi`).

**During execution**:
-   The application will display the live video feed with the predicted class and confidence level.
-   Press 'q' to quit the application.

## Project Components

### 1. Dataset and Preprocessing (`preprocess_data.py`)
-   **Dataset**: The dataset consists of about 100 images per class for four classes.
-   **Preprocessing**: The `preprocess_data.py` script performs the following steps:
    -   Loads images from the dataset directory.
    -   Resizes all images to a uniform size of `(224, 224)`.
    -   Normalizes pixel values to the range `[0, 1]`.
    -   Saves the processed images (`X`) and their corresponding labels (`y`) into `.npy` files.

### 2. CNN Model Architecture (`train_model.py`)
The CNN model is a sequential model with the following architecture:

| Layer Type      | Details                                    |
| --------------- | ------------------------------------------ |
| `Conv2D`        | 32 filters, (3, 3) kernel, ReLU activation |
| `MaxPooling2D`  | (2, 2) pool size                           |
| `Conv2D`        | 64 filters, (3, 3) kernel, ReLU activation |
| `MaxPooling2D`  | (2, 2) pool size                           |
| `Conv2D`        | 64 filters, (3, 3) kernel, ReLU activation |
| `MaxPooling2D`  | (2, 2) pool size                           |
| `Flatten`       | Flattens the output to a 1D vector         |
| `Dense`         | 64 units, ReLU activation                  |
| `Dropout`       | 0.5 rate for regularization                |
| `Dense` (Output)| 4 units (one for each class), Softmax activation |

-   **Training**: The model is compiled with the `Adam` optimizer and `categorical_crossentropy` loss function. It is trained for 50 epochs.

### 3. Real-time Classification (`LiveClassifier.py`)
The `LiveClassifier.py` script performs the following steps in a loop:
1.  Captures a frame from the video source.
2.  Preprocesses the frame to match the model's input requirements (resizing, normalization).
3.  Passes the frame to the `model.predict()` method to get the classification results.
4.  Determines the class with the highest probability.
5.  Overlays the predicted class label and confidence score on the frame.
6.  Calculates and displays the Frames Per Second (FPS).
7.  Displays the processed frame and writes it to an output video file.

## Files in the Repository

-   **`LiveClassifier.py`**: The main script for running the real-time object classification.
-   **`preprocess_data.py`**: A script to preprocess the image dataset before training.
-   **`train_model.py`**: A script to build, train, and save the CNN model.
-   **`live_classifier_model.h5`**: The pre-trained Keras model file.

## Video Demonstration

[https://youtube.com/shorts/_oR95TbGVP4?feature=share](https://youtube.com/shorts/_oR95TbGVP4?feature=share)
