# webcam_detector.py
import cv2
import torch
import numpy as np

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
    model.conf = 0.5  # Confidence threshold
    return model

def process_frame(frame, model):
    results = model(frame)
    
    # Draw bounding boxes
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det.tolist()
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def main():
    model = load_model()
    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, model)
        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()