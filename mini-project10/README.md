# Mini-Project 10: U-Net Model for Vehicle Segmentation

## Overview

This project implements a U-Net, a type of Convolutional Neural Network (CNN), for vehicle segmentation. The goal is to take an image of a vehicle and produce a binary mask that highlights the pixels belonging to the vehicle. The project includes a complete pipeline for data generation, model training, and result visualization.

## Workflow

1.  **Dataset Generation**: A custom dataset is created using several scripts. Images are downloaded, and then YOLO is used to generate segmentation masks for the vehicles in them.
2.  **Configuration**: All parameters for the project (file paths, model hyperparameters, etc.) are managed in the `config/config.yaml` file.
3.  **Training**: The `main.py` script orchestrates the training process. It loads the data, builds the U-Net model, and trains it on the dataset.
4.  **Evaluation & Visualization**: The trained model is used to predict masks on test images, and the results are visualized.

## Project Structure

The project follows a structured layout with separate directories for data, source code, configuration, and results.

```
mini-project10/
├── data/
├── src/
├── config/
├── results/
└── ...
```

## How to Run

1.  **Setup Environment**:
    ```bash
    # (Optional) Create a virtual environment
    python -m venv venv
    source venv/bin/activate
    # Install dependencies
    pip install -r requirements.txt
    ```

2.  **Dataset Generation**:
    Run the following scripts in order to generate the dataset.
    ```bash
    python generate_data.py
    python generate_masks.py
    python clean_data.py
    python generate_labels.py
    ```

3.  **Model Training**:
    Once the dataset is ready, run the main script to start training.
    ```bash
    python main.py
    ```

## Dataset Creation Scripts

-   `generate_data.py`: Downloads vehicle images using Bing Image Downloader.
-   `generate_masks.py`: Uses a pre-trained YOLO model to detect vehicles in the downloaded images and generates binary segmentation masks for them.
-   `clean_data.py`: A utility script to clean up the dataset (e.g., remove images where no vehicles were detected).
-   `generate_labels.py`: Generates YOLO format labels from the masks.

## Implementation Details

### Configuration (`config/config.yaml`)
This file stores all the important parameters for the project, such as file paths, image dimensions, batch size, learning rate, and data augmentation settings. This makes it easy to experiment with different configurations without changing the source code.

### Data Pipeline (`src/data/make_dataset.py`)
This script handles all aspects of data preparation for the model:
-   **Loading**: Loads images and their corresponding masks from the `data/raw` directory.
-   **Preprocessing**: Resizes images and masks to the required input dimensions and normalizes image pixel values to be between 0 and 1.
-   **Data Splitting**: Splits the dataset into training and testing sets.
-   **Data Augmentation**: Creates a data generator (`create_data_generator`) that applies random transformations (rotation, shifting, zooming, flipping) to the training data. This helps the model generalize better and prevents overfitting.

### Model Architecture (`src/models/models.py`)
This script defines the U-Net model architecture.
-   **U-Net**: The U-Net model consists of an **encoder** (contracting path) and a **decoder** (expansive path).
    -   The **encoder** uses a series of convolutional and max-pooling layers to capture the context of the image.
    -   The **decoder** uses upsampling (transposed convolutions) to gradually reconstruct the segmentation map.
    -   **Skip Connections** are a key feature of U-Net. They connect the output of the encoder layers to the corresponding layers in the decoder, which helps the model recover fine-grained details that might be lost during downsampling.
-   **Loss Functions**:
    -   **Dice Loss (`dice_loss`)**: This loss function is well-suited for image segmentation tasks, especially with imbalanced classes (e.g., more background pixels than object pixels). It is based on the Dice coefficient, which measures the overlap between the predicted and true masks.
    -   **BCE Dice Loss (`bce_dice_loss`)**: A combination of Binary Cross-Entropy and Dice loss, which can lead to more stable training.
-   **Metrics**:
    -   **IoU Score (`iou_score`)**: Intersection over Union (also known as the Jaccard index) is another common metric for segmentation tasks.

### Model Training (`src/models/train_model.py`)
This script manages the model training process.
-   The `ModelTrainer` class builds and compiles the U-Net model.
-   It uses several Keras **callbacks** to improve the training process:
    -   `ModelCheckpoint`: Saves the best version of the model based on the validation Dice coefficient.
    -   `ReduceLROnPlateau`: Reduces the learning rate if the validation loss stops improving.
    -   `EarlyStopping`: Stops the training process early if the validation metric does not improve for a certain number of epochs, preventing overfitting.

### Visualization (`src/visualization/visualize.py`)
This script provides utilities for visualizing the results, such as plotting the training history (loss and accuracy over epochs) and displaying the predicted masks alongside the original images and ground truth masks.

### Model and Data Links
The generated data and pre-trained models can be found [here](https://northeastern-my.sharepoint.com/:f:/r/personal/ambulkar_m_northeastern_edu/Documents/mini-project10?csf=1&web=1&e=eScXmp).