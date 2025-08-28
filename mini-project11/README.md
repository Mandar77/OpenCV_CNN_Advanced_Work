# Mini-Project 11: Real-Time Object Detection and Tracking

## Overview

This project demonstrates a real-time system that combines object detection using YOLOv8 with motion tracking using Lucas-Kanade optical flow. The application processes a video feed from a webcam or a file, draws bounding boxes around detected objects, and visualizes motion by drawing trails for prominent features in the scene.

## Features

-   **Object Detection**: Uses a pre-trained YOLOv8n model to detect and track objects in real-time.
-   **Motion Tracking**: Implements Lucas-Kanade optical flow to track the movement of keypoints between frames.
-   **Combined Visualization**: Overlays both the YOLO detection bounding boxes and the optical flow tracks onto the video feed.
-   **Video Input**: Can process video from a live webcam or a pre-recorded video file.
-   **Video Output**: Saves the processed video with all visualizations to `output.avi`.

## Setup Instructions

1.  **Clone the repository** (if you haven't already).
2.  **Install required libraries**:
    ```bash
    pip install opencv-python ultralytics numpy
    ```
    *Note*: The `ultralytics` package provides the YOLOv8 implementation. The line `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'` is included in the script to avoid potential issues with conflicting libraries on some systems.

## How to Run

-   **To use the default webcam**:
    ```bash
    python WebCamSave.py
    ```
-   **To use a video file**:
    ```bash
    python WebCamSave.py -f path/to/your/video.mp4
    ```
-   Press 'q' to quit the application. The output will be saved as `output.avi`.

## How It Works

The script processes the video frame by frame in a main loop, performing both detection and tracking in each iteration.

1.  **Initialization**:
    -   A YOLOv8n model is loaded.
    -   Video capture and a video writer are initialized.
    -   Parameters for the Lucas-Kanade optical flow algorithm are set.

2.  **Main Loop**:
    -   **YOLO Detection**: The current frame is passed to the `yolo.track()` method. This returns the bounding boxes, class labels, and confidence scores for the detected objects.
    -   **Optical Flow Calculation**:
        -   The frame is converted to grayscale.
        -   In the first frame, a set of strong corners to track is detected using `cv2.goodFeaturesToTrack`.
        -   In subsequent frames, `cv2.calcOpticalFlowPyrLK` is used to calculate the new positions of these points from the previous frame to the current one.
    -   **Visualization**:
        -   The YOLO bounding boxes and labels are drawn on the frame in blue.
        -   The optical flow tracks (lines showing the movement of keypoints) are drawn on a mask, which is then overlaid on the frame. The current position of each tracked point is marked with a red circle.
    -   **Output**: The combined frame is displayed on the screen and written to the output video file.

## Implementation Details

### Object Detection with YOLOv8
-   The script uses the `ultralytics` library, which is the official implementation of YOLOv8.
-   `yolo = YOLO('yolov8n.pt')` loads the smallest, fastest pre-trained YOLOv8 model.
-   `results = yolo.track(frame, persist=True)[0]` performs detection and tracking. The `persist=True` argument tells the tracker to remember the tracks from the previous frame.

### Motion Tracking with Optical Flow
-   **Lucas-Kanade Method**: This is a sparse optical flow method, meaning it tracks a sparse set of feature points (not every pixel).
-   **`cv2.goodFeaturesToTrack`**: This function is used to find prominent corners in the image, which are good features to track.
-   **`cv2.calcOpticalFlowPyrLK`**: This is the core function for calculating the optical flow. It uses image pyramids (`Pyr`) to handle larger motions.

**Note on the combination**: In this implementation, the YOLO detection and optical flow are performed mostly independently. YOLO detects and tracks objects, while optical flow tracks the motion of generic feature points across the entire frame. The visualizations are then combined. A more advanced implementation might use the YOLO detections to initialize the points for optical flow, allowing for more targeted tracking of specific objects.

## Video Demonstration

A video demonstration of the project can be found [here](https://youtu.be/iYdC2y10cRQ).