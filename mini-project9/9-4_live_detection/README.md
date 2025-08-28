# Project 9-4: Live Detection of Stop Signs and Traffic Lights with YOLOv5

## Overview

This project uses a custom-trained YOLOv5 model to perform real-time detection of "Stop Signs" and "Traffic Lights" from a live webcam feed or a video file. The project covers the entire workflow from setting up the dataset and training the model to running live inference.

## Workflow

1.  **Setup**: Clone the YOLOv5 repository and organize the custom dataset.
2.  **Configuration**: Create a `data.yaml` file to define the dataset paths and class names.
3.  **Training**: Train the YOLOv5 model on the custom dataset. The `train.py` script is a helper for this process.
4.  **Inference**: Use the `WebCamSave.py` script to run the trained model on a live video feed or a video file.

## Setup and Configuration

1.  **Clone YOLOv5 Repository**:
    First, you need to clone the official YOLOv5 repository, which contains the training and detection scripts.
    ```bash
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    pip install -r requirements.txt
    ```

2.  **Download and Organize Dataset**:
    -   Download the custom dataset of stop signs and traffic lights from the link in the "Model and Data Access" section.
    -   Organize your dataset into `train`, `valid`, and `test` sets, with `images` and `labels` subdirectories for each, as expected by YOLOv5.

3.  **Configure `data.yaml`**:
    Create a `data.yaml` file and place it in the `yolov5` directory. This file tells the training script where to find the data and what the classes are.
    ```yaml
    # The number of classes in your dataset
    nc: 2

    # The names of your classes
    names: ['Stop Sign', 'Traffic Light']

    # Paths to your training and validation data
    # These should be relative to the yolov5 directory
    train: ../path/to/your/dataset/train/images
    val: ../path/to/your/dataset/valid/images
    ```

## Scripts in this Directory

-   **`train.py`**: A helper script that configures and launches the YOLOv5 training process. It sets hyperparameters and paths before calling the official `yolov5/train.py` script. **Note**: This script may contain hardcoded local paths and might need to be adapted to your directory structure.
-   **`WebCamSave.py`**: The main script for running inference. It loads the custom-trained model (`best.pt`) and performs real-time detection on a video source.

## How to Train the Model

To train the model, you run the `train.py` script from the YOLOv5 directory. The `train.py` script in this project is a wrapper that sets up the command for you. A more general way to run the training is:

```bash
python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov5s.pt --cache
```

-   `--img`: Input image size.
-   `--batch`: Batch size.
-   `--epochs`: Number of training epochs.
-   `--data`: Path to your `data.yaml` file.
-   `--weights`: The pre-trained model to start from (e.g., `yolov5s.pt` for transfer learning).

After training, the best model weights will be saved as `runs/train/exp/weights/best.pt`. You should move this file to the `9-4_live_detection` directory.

## How to Run Live Detection

The `WebCamSave.py` script uses the trained `best.pt` model to perform detection.

-   **To run on a live webcam feed**:
    ```bash
    python WebCamSave.py
    ```
-   **To run on a video file and save the output**:
    ```bash
    python WebCamSave.py -f path/to/your/video.mp4 -o path/to/output.avi
    ```
    -   `-f`: Path to the input video file.
    -   `-o`: Path to save the output video with detections.

## Model and Data Access

-   **Dataset**: The training dataset can be downloaded from [this link](https://northeastern-my.sharepoint.com/:f:/r/personal/ambulkar_m_northeastern_edu/Documents/mini-project9?csf=1&web=1&e=5Zhica).
-   **Testing Videos**: Real-world driving videos for testing can be accessed [here](https://northeastern-my.sharepoint.com/:v:/g/personal/ambulkar_m_northeastern_edu/ERFRsfHtIcVHig-FeHjv7TcBmbto97ri10JkdTsKMZ-WcA?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&email=jorwekar.h%40northeastern.edu&e=x1ciZO).
