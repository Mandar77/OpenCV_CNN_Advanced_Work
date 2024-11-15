# Group 1: Stop Sign and Traffic Signal Live Detection Using YOLOv5

This project aims to develop an object detection system using a YOLOv5 model to recognize and classify "Stop Sign" and "Traffic Light" objects in real-time.

## Dataset Information

### 1. Training Dataset

The training dataset consists of labeled images for "Stop Sign" and "Traffic Light" classes.

- **Classes**:
  - 1383 training images and corresponding labels
  - 100 testing images and corresponding labels
  - 300 validation images and corresponding labels
  - Somewhat equally distributed into both "Stop Signs" and Traffic Signals"

- **Annotation Format**: Each image is annotated using YOLO format, where each `.txt` file contains the coordinates and class labels for objects in the image.

- **Source**: The training dataset can be downloaded [[here](https://northeastern-my.sharepoint.com/:f:/r/personal/ambulkar_m_northeastern_edu/Documents/mini-project9?csf=1&web=1&e=5Zhica)].

### 2. Testing Dataset

The testing dataset consists of two real-world driving videos captured on city streets, simulating real-time scenarios for model evaluation.

- **Dataset**: Video of real driving scenario.
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
  - **Optimizer**: Adam.
  - **Patience**: 50
  - **Save-Period**: 10
  - **Workers**: 8 (Can be adjusted according to CPU cores)

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

- Modify `data.yaml` in YOLOv5 repository as follows:
  
  ```
  names:
  - Stop Sign
  - Traffic Light
  nc: 2
  test: D:\KhouryGithub\CS5330_FA24_Group1\mini-project9\yolov5\datasets\test\images
  train: D:\KhouryGithub\CS5330_FA24_Group1\mini-project9\yolov5\datasets\train\images
  val: D:\KhouryGithub\CS5330_FA24_Group1\mini-project9\yolov5\datasets\valid\images

  ```

4. Train the YOLOv5 model:

```
python 9-4_live_detection\train.py
```

5. Run detection and save results:

```
python WebCamSave.py -o output_video.avi
python WebCamSave.py -f real_dataset/test1.mp4 -o test1_output.avi
```

These results highlight the model’s effectiveness in recognizing both "Stop Sign" and "Traffic Light" objects, supporting its potential application in real-time autonomous driving systems.
