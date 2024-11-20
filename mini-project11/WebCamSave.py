import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import numpy as np
import argparse
from ultralytics import YOLO

# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
args = parser.parse_args()

# Initialize YOLO model
yolo = YOLO('yolov8n.pt')

# Initialize video capture
if args.file:
    cap = cv2.VideoCapture(args.file)
else:
    cap = cv2.VideoCapture(0)  # Use default camera

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

# Initialize variables for optical flow
old_frame = None
old_gray = None
p0 = None
mask = np.zeros((height, width, 3), dtype=np.uint8)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLO detection
    results = yolo.track(frame, persist=True)[0]
    
    # Convert frame to grayscale for optical flow
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if old_frame is None:
        old_frame = frame.copy()
        old_gray = frame_gray.copy()
        # Initialize points to track (e.g., corners)
        p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    else:
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            # Draw tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            # Update points
            p0 = good_new.reshape(-1, 1, 2)

        # Combine frame with optical flow visualization
        img = cv2.add(frame, mask)

        # Draw YOLO detections
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            class_id = int(box.cls.item())
            conf = box.conf.item()
            label = f"{results.names[class_id]} {conf:.2f}"
            
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Object Detection and Tracking', img)

        # Write frame to output video
        out.write(img)

        # Update previous frame and gray image
        old_frame = frame.copy()
        old_gray = frame_gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()