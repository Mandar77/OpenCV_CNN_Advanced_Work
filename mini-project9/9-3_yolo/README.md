# Project 9-3: Object Detection with YOLOv3

## Overview

This project explores object detection using YOLOv3 (You Only Look Once, version 3), a popular real-time object detection algorithm. The project includes two main scripts: a generic YOLOv3 detector and a specialized application that detects specific objects ("cup" and "banana") from a webcam and automatically records video clips when they are found.

## Features

-   **YOLOv3 Integration**: Uses a pre-trained YOLOv3 model to detect objects from the COCO dataset.
-   **Generic Object Detector**: A script (`object_detection_yolo.py`) to run YOLOv3 on image or video files.
-   **Specialized Webcam Detector**: A script (`WebCamSave_Yolo.py`) that:
    -   Detects specific target classes ("cup" and "banana") from a live webcam feed.
    -   Automatically records a 5-second video clip when a target object is detected.
    -   Is structured with classes for the detector and video recorder, making it easy to customize.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install opencv-python numpy
    ```
2.  **Download YOLOv3 Model Files**:
    You need the YOLOv3 weights, configuration file, and class names. A shell script `getModels.sh` is provided to download these files. Run it from your terminal:
    ```bash
    bash getModels.sh
    ```
    This will download `yolov3.weights`, `yolov3.cfg`, and `coco.names`.

## Scripts in this Directory

-   **`object_detection_yolo.py`**: A generic, command-line based script for running YOLOv3 object detection. It is based on the official OpenCV example and can be used to process both image and video files.
-   **`WebCamSave_Yolo.py`**: A custom application that uses the webcam to detect specific objects ("cup" and "banana") and automatically saves a 5-second video clip when they are detected.

## How to Run

### 1. Generic Object Detector (`object_detection_yolo.py`)
You can use this script to run YOLOv3 on a video or image file.

-   **For a video file**:
    ```bash
    python object_detection_yolo.py --video=your_video.mp4
    ```
-   **For an image file**:
    ```bash
    python object_detection_yolo.py --image=your_image.jpg
    ```

### 2. Webcam Detector with Auto-Recording (`WebCamSave_Yolo.py`)
This script is designed to be run without command-line arguments.

```bash
python WebCamSave_Yolo.py
```
-   The application will open a window showing the webcam feed.
-   If a "cup" or "banana" is detected, it will start recording a 5-second video.
-   The recorded videos will be saved as `DetectedObject_1.mp4`, `DetectedObject_2.mp4`, and so on.
-   Press 'q' to quit.

## Implementation Details (`WebCamSave_Yolo.py`)

The `WebCamSave_Yolo.py` script is structured into two main classes:

### `ObjectDetector`
-   This class encapsulates all the logic related to the YOLOv3 model.
-   **`__init__`**: Loads the YOLOv3 network, class names, and sets the target classes.
-   **`detect_objects`**: Takes a frame as input, runs it through the network, and returns a list of detected objects.
-   **`process_detections`**: Applies non-maximum suppression (NMS) to filter out weak and overlapping bounding boxes.
-   **`draw_prediction`**: Draws the bounding boxes and labels on the frame.

### `VideoRecorder`
-   This class handles the logic for recording video clips.
-   **`start_recording`**: Sets a flag to start recording and initializes a list to store frames.
-   **`add_frame`**: Adds the current frame to the list if recording is active.
-   **`stop_recording`**: Saves the stored frames to a `.mp4` video file and resets the state.
-   **`should_stop`**: Checks if the 5-second recording duration has passed.

### Main Logic
The main part of the script runs a loop that:
1.  Reads a frame from the webcam.
2.  Calls the `detector.detect_objects` method.
3.  Checks if any of the detected objects are in the `target_classes` list.
4.  If a target object is found and recording is not already in progress, it calls `recorder.start_recording()`.
5.  If the recorder has been running for 5 seconds, it calls `recorder.stop_recording()`.