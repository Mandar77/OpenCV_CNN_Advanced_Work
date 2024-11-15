# Group 1: Stop Sign and Traffic Signal Live Detection Using YOLOv5

This project aims to develop an object detection system using a YOLOv5 model to recognize and classify "Stop Sign" and "Traffic Light" objects in real-time.

## Dataset Information

### 1. Training Dataset

The training dataset consists of labeled images for "Stop Sign" and "Traffic Light" classes.

- **Classes**:
  - **Traffic Light**: 1,402 images with bounding box annotations in `.txt` format.
  - **Stop Sign**: 715 images with bounding box annotations in `.txt` format.
- **Annotation Format**: Each image is annotated using YOLO format, where each `.txt` file contains the coordinates and class labels for objects in the image.
- **Source**: The training dataset can be downloaded [[here](https://northeastern-my.sharepoint.com/:f:/r/personal/ambulkar_m_northeastern_edu/Documents/mini-project9?csf=1&web=1&e=5Zhica)].

### 2. Testing Dataset

The testing dataset consists of two real-world driving videos captured on city streets, simulating real-time scenarios for model evaluation.

- **Dataset**: Two video files of real driving scenarios in urban settings.
- **Source**: The testing dataset can be accessed [here](https://northeastern-my.sharepoint.com/:v:/g/personal/ambulkar_m_northeastern_edu/ERFRsfHtIcVHig-FeHjv7TcBmbto97ri10JkdTsKMZ-WcA?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&email=jorwekar.h%40northeastern.edu&e=x1ciZO).

### 3. Output Data

- **Dataset**: The output dataset contains annotated test results after running the trained model on the driving videos.
- **Source**: The output data can be downloaded [add link here].

---

## Model Training

The model was trained using YOLOv5, a highly efficient object detection algorithm for real-time detection. Below are the key steps and settings in the training process.

### 1. Preprocessing

- The images were resized to ensure consistency across the dataset.
- YOLOv5 annotation format was used, where each object in the image is labeled with its class and bounding box coordinates in a `.txt` file.

### 2. Training Process

- **Transfer Learning**: YOLOv5's pretrained weights (`yolov5s.pt`) served as the base model to speed up training and improve accuracy.
- **Model Architecture**: YOLOv5 small model (`yolov5s`) was chosen for its balance between speed and accuracy.
- **Hyperparameters**:
  - **Epochs**: 100 epochs for model convergence.
  - **Batch Size**: 16 images per batch.
  - **Learning Rate**: 0.01 (default for YOLOv5).
  - **Optimizer**: Stochastic Gradient Descent (SGD) with momentum.
- **Early Stopping**: Not applied here, but could be considered to halt training if no validation mAP improvement is observed.

### 3. Steps to Train the Model

1. **Download Dataset**

   - Download and unzip the training dataset from the link above.

2. **Clone the YOLOv5 Repository**

   ```
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   pip install -r requirements.txt
   ```
3. Configure Dataset in YOLOv5:

- Modify `traffic.yaml` in YOLOv5 repository as follows:
  ```
  # Directory paths
  train: ../custom_dataset/train/images
  val: ../custom_dataset/train/images

  # Number of classes
  nc: 2

  # Class names
  names: ["Stop Sign", "Traffic Light"]
  ```

4. Train the YOLOv5 model:

```
python train.py --img 640 --batch 16 --epochs 100 --data traffic.yaml --weights yolov5s.pt --name traffic_detection
```

5. Run detection and save results:

```
python WebCamSave.py -o output_video.avi
python WebCamSave.py -f real_dataset/test1.mp4 -o test1_output.avi
```

## Model Evaluation

After training, the model was evaluated on two real-world driving videos to assess its accuracy and reliability in detecting "Stop Sign" and "Traffic Light" objects. The evaluation focused on several performance metrics critical for object detection.

- **Evaluation Metrics**:
  - **Precision (P)**: Represents the accuracy of the positive predictions, reflecting how often the detected objects are correctly classified.
  - **Recall (R)**: Measures the model’s ability to identify all relevant instances of the target classes.
  - **mAP@0.5**: Mean Average Precision at a 0.5 Intersection over Union (IoU) threshold, providing a comprehensive measure of the model’s precision and recall across all detections.
  - **mAP@0.5:0.95**: Mean Average Precision over IoU thresholds from 0.5 to 0.95, in increments of 0.05, offering a more rigorous assessment of model performance by evaluating its precision and recall under varied IoU conditions.

- **Results Summary**:
  - **Overall mAP**: The model achieved an overall mAP of 0.991, demonstrating a high level of accuracy in detecting the specified traffic elements.
  - **Stop Sign Detection**:
    - **Precision**: 0.993
    - **Recall**: 0.999
    - Interpretation: The model showed excellent performance in identifying stop signs, with nearly perfect recall and high precision, ensuring few false positives.
  - **Traffic Light Detection**:
    - **Precision**: 0.989
    - **Recall**: 0.977
    - Interpretation: Detection of traffic lights was also strong, although slightly lower than for stop signs due to potential variations in lighting and distance in the video footage.

These results highlight the model’s effectiveness in recognizing both "Stop Sign" and "Traffic Light" objects, supporting its potential application in real-time autonomous driving systems.
