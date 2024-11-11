# import the necessary packages
import cv2
import numpy as np
import time
import torch
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name")

args = parser.parse_args()

# Check if the file argument is provided, otherwise use the camera
if args.file:
    vs = cv2.VideoCapture(args.file)
else:
    vs = cv2.VideoCapture(0)  # 0 is the default camera

time.sleep(2.0)

# Get the default resolutions
width  = int(vs.get(3))
height = int(vs.get(4))

# Define the codec and create a VideoWriter object
out_filename = args.out
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height), True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)  # Load custom model
model.eval()  # Set the model to evaluation mode

# loop over the frames from the video stream
while True:
    # grab the frame from video stream
    ret, frame = vs.read()
    if not ret:
        break

    # Convert frame to the format expected by YOLO model
    results = model(frame)  # Run inference on the frame

    # Process the results
    for detection in results.xyxy[0]:  # results.xyxy[0] gives bounding boxes in [x1, y1, x2, y2, confidence, class]
        x1, y1, x2, y2, confidence, cls = detection
        label = model.names[int(cls)]  # Get class label

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video file
    if args.out:
        out.write(frame)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Release the video capture object
vs.release()
out.release()
cv2.destroyAllWindows()