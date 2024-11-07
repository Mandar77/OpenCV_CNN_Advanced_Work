# Live Detector: Object Recognition for Autonomous Cars

This project implements a real-time object detection system using YOLOv5 to recognize "Stop Sign" and "Traffic Signal" objects.

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Clone YOLOv5: `git clone https://github.com/ultralytics/yolov5`
6. Install YOLOv5 dependencies: `cd yolov5 && pip install -r requirements.txt && cd ..`

## Usage

1. Prepare your custom dataset and update `custom_data.yaml`
2. Train the model: `python train_model.py`
3. Run the live detector: `python webcam_detector.py`

## Dataset

- Number of images: 200+ (100+ per class)
- Classes: "Stop Sign", "Traffic Signal"
- Preprocessing: Resized to 640x640, augmented with rotations and flips

## Model Training

- Base model: YOLOv5s
- Transfer learning: Fine-tuned on custom dataset
- Epochs: 100
- Batch size: 16