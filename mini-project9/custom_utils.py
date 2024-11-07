# utils.py

import cv2
import torch
import numpy as np

def load_model(model_name):
    model = torch.hub.load('ultralytics/yolov5', model_name)
    return model

def process_frame(frame, model):
    # Perform inference
    results = model(frame)

    # Draw bounding boxes and labels
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf > 0.5:  # Confidence threshold
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame