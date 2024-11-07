import os

# Set environment variable to avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

def train_yolo():
    # Load a pretrained YOLOv5 model
    model = YOLO('yolov5s.pt')
    
    # Train the model
    results = model.train(data='custom_data.yaml', epochs=100, imgsz=640)
    
    # Export the trained model
    model.export()

if __name__ == "__main__":
    train_yolo()