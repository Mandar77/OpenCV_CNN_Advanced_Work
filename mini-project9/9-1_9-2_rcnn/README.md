# Project 9-1 & 9-2: Multi-class Detection using R-CNN

## Overview

This project implements a multi-class object detector using a simplified R-CNN (Regions with CNN features) approach. The goal is to detect two classes of objects: "remotes" and "airplanes". The project is divided into two main parts:
-   **9-1**: Developing and training the multi-class R-CNN model.
-   **9-2**: "Optimizing" the model for better performance, which in this case involves retraining the model, and testing it in a real-time scenario.

The implementation uses a two-stage training process. First, a base model (VGG16) is fine-tuned to distinguish object proposals from the background. Then, a second model is trained on top of the base model's features to perform the final multi-class classification.

## Workflow

1.  **Data Preparation**:
    -   Images for "remotes" and "airplanes" are collected.
    -   Bounding box annotations are created for these images (e.g., using a tool like LabelMe).
    -   The `json_to_csv.py` script is used to convert the annotations from JSON format to a CSV file.
2.  **Model Training (`rcnn-multi.py` or `rcnn-light.py`)**:
    -   The script generates region proposals for each image.
    -   It creates a training dataset by labeling proposals as "positive" (object) or "negative" (background) based on their Intersection over Union (IoU) with the ground truth boxes.
    -   It trains the two-stage model on this data.
3.  **Evaluation**:
    -   The trained model is tested on sample images to generate detection results.
    -   The `WebCamSave-rcnn.py` script is used to evaluate the model's performance on a live webcam feed.

## Implementation Details

### 1. Region Proposal
-   A simple sliding window approach (`generate_proposals`) is used to generate region proposals. This is a simplified alternative to the Selective Search algorithm used in the original R-CNN paper.

### 2. Data Generation
-   For each image, positive and negative training samples are generated:
    -   **Positive samples**: Region proposals with an IoU > 0.5 with a ground truth box.
    -   **Negative samples**: Region proposals with an IoU < 0.1 with any ground truth box.
-   This data is used to train the base model.
-   A separate dataset is created for the final classifier, containing the ground truth objects and some negative samples.

### 3. Two-Stage Model Training
-   **Base Model**: A VGG16 model, pre-trained on ImageNet, is used as the base. The top layers are replaced with a single dense layer with a sigmoid activation function. This model is trained as a binary classifier to distinguish objects from the background.
-   **Final Model**: The output of the base model's global average pooling layer is fed into a new dense layer with 2 output units. This model is compiled with a hinge loss, making it act like a linear SVM, to classify the objects into "remote" or "airplane".

### 4. Non-Maximum Suppression (NMS)
-   After detection, the `apply_non_max_suppression` function is used to filter out redundant, overlapping bounding boxes, keeping only the ones with the highest confidence scores.

## Scripts in this Directory

-   `json_to_csv.py`: A utility script to convert bounding box annotations from LabelMe's JSON format to a CSV file required by the training scripts.
-   `rcnn-multi.py`: The main script for training the multi-class R-CNN model. It saves the trained model as `multi_class_detector.h5`.
-   `rcnn-light.py`: This script is identical to `rcnn-multi.py` but saves the model as `multi_class_detector_light.h5`. It is intended for training the "optimized" or retrained version of the model.
-   `WebCamSave-rcnn.py`: A script to run the trained model on a live webcam feed to test its real-time performance.
-   `test_rcnn.py`: A script to evaluate the performance of the trained model.

## How to Run

### 1. Prepare the Data
1.  Place your images in the `airplanes` and `remotes` directories.
2.  Annotate your images and generate CSV files (`airplanes_annotations.csv`, `remotes_annotations.csv`) using `json_to_csv.py`. The CSV should have columns: `image_path`, `x_min`, `y_min`, `x_max`, `y_max`.

### 2. Train the Model
-   To train the standard model, run:
    ```bash
    python rcnn-multi.py
    ```
-   To train the "light" model, run:
    ```bash
    python rcnn-light.py
    ```
    This will save the trained models as `.h5` files.

### 3. Test the Model
-   To test the model on a live webcam feed, you will need to modify `WebCamSave-rcnn.py` to load your trained model and run it.
-   To evaluate the model's performance, use the `test_rcnn.py` script.

## Model and Data Storage
-   The datasets and pre-trained models are large and can be accessed from the cloud storage link provided in the original `README.md`.