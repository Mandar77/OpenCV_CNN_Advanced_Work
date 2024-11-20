# CS5330 Group 1 - Mini-Project 11: Follow Everything

## Project Overview

This project implements a real-time object detection and tracking system using a webcam. It combines YOLO (You Only Look Once) for object detection with an optical flow algorithm for tracking. The application detects objects using YOLO and then tracks their movement across frames using optical flow, allowing it to follow detected objects as they move.

## Team Members
1. Cen Chang
2. Harshal Jorwekar
3. Mandar Ambulkar

### Video Demonstration
A video demonstration of the project can be found [here](https://youtu.be/iYdC2y10cRQ). The video first plays the input video and then the output of the program.

## Setup Instructions
- Clone the repository:

```
git clone https://github.khoury.northeastern.edu/mandar07/CS5330_FA24_Group1.git
cd CS5330_FA24_Group1/mini-project11
```

- Install required libraries:

```
pip install opencv-python ultralytics numpy
```

## Usage:

- Run the main script:

```
python WebCamSave.py [-f VIDEO_FILE]
```

- Use -f VIDEO_FILE to specify an input video file. If not provided, the script will use the default webcam.
- Press 'q' to quit the application.

## Features

- Object Detection: Uses YOLO to detect objects in real-time from the webcam feed.
- Object Tracking: Implements optical flow to track detected objects across frames.
- Multiple Object Handling: Can detect and track multiple objects simultaneously.
- Video Output: Saves the processed video with detection and tracking visualizations as 'output.avi'.

## Implementation Details

- YOLO Model: We use YOLOv8n for object detection.
- Optical Flow Algorithm: Lucas-Kanade method is used for tracking objects between frames.
- Target Objects: The system can detect and track multiple objects, with a focus on [specify your target objects, e.g., persons, vehicles].

## Project Structure

- WebCamSave.py: Main script combining YOLO detection and optical flow tracking.
- live_opticalflow.py: Reference implementation of optical flow (not directly used in the final solution).
- README.md: This file, containing project documentation.

### Work Breakdown

1. Mandar Ambulkar:
    - Implement YOLO object detection
    - Integrate YOLO with WebCamSave.py
    - Write README section on object detection
2. Harshal Jorwekar:
    - Implement optical flow tracking
    - Integrate tracking with YOLO detection
    - Handle multiple object tracking
3. Cen Chang:
    - Set up development environment
    - Handle video input/output
    - Record demonstration video
    - Complete remaining README sections

4. Shared Responsibilities:
    - Testing and debugging
    - Performance optimization
    - Final documentation review