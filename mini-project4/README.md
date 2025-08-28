# Mini-Project 4: Dual Webcam Feature Matching

## Overview

This Python script demonstrates real-time feature matching between two webcam feeds using OpenCV. It captures video from two separate webcams, detects keypoints and computes descriptors using the SIFT (Scale-Invariant Feature Transform) algorithm, and then matches these features using a Brute-Force matcher. The results, including the matched features, a matching score, and the frames per second (FPS), are displayed in real-time.

## Features

- **Dual Webcam Support**: Simultaneously captures and processes video from two webcams.
- **SIFT Feature Detection**: Utilizes SIFT for robust detection of keypoints and computation of feature descriptors.
- **Brute-Force Matching**: Employs a Brute-Force matcher to find the best matches between the features of the two video feeds.
- **Real-time Visualization**: Displays the two video feeds side-by-side with lines connecting the matched features.
- **Performance Metrics**: 
  - **Matching Score**: Calculates and displays a score based on the distances of the matched features. A lower score indicates a better match.
  - **FPS Counter**: Shows the real-time Frames Per Second (FPS) to indicate processing speed.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- OpenCV Contrib (`opencv-contrib-python` for SIFT)
- Two webcams connected to the computer.

## Installation

1.  Ensure Python 3.x is installed.
2.  Install the required libraries:
    ```bash
    pip install opencv-python opencv-contrib-python
    ```

## How to Run

1.  Connect two webcams to your computer. Note that the script uses camera indices 0 and 1. If your cameras have different indices, you may need to change `cv2.VideoCapture(0)` and `cv2.VideoCapture(1)`.
2.  Run the script from your terminal:
    ```bash
    python mini-project4.py
    ```
3.  A window titled "Feature Matching with Dual Webcam" will open, showing the combined feed from both webcams with lines drawn between matched features.
4.  Press 'q' to exit the program.

## Implementation Details

### 1. Webcam Initialization
-   Two `cv2.VideoCapture` objects are created to capture video from two different webcams (indices 0 and 1).

### 2. Feature Detection and Matching
-   **SIFT (Scale-Invariant Feature Transform)**: A `cv2.SIFT_create()` object is initialized. In the main loop, for each frame, SIFT is used to detect keypoints and compute their descriptors.
-   **Brute-Force (BF) Matcher**: A `cv2.BFMatcher` with `cv2.NORM_L2` (Euclidean distance) is created. This matcher compares each descriptor from the first webcam's frame with all descriptors from the second webcam's frame and finds the closest match.
-   **Match Sorting**: The matches are sorted based on their `distance` attribute in ascending order. The lower the distance, the better the match.

### 3. Visualization
-   `cv2.drawMatches()`: This function is used to draw the top 50 matches. It places the two frames side-by-side and draws lines between the corresponding keypoints of the matched features.

### 4. Performance Metrics
-   **Matching Score**: The script calculates a `matching_score` by summing the distances of the top 50 matches. This score is displayed on the output frame.
-   **FPS**: The Frames Per Second are calculated by taking the reciprocal of the time difference between the processing of consecutive frames. This is also displayed on the frame.

### 5. Main Loop
-   The script runs a `while True` loop that continuously performs the following steps:
    1.  Captures a frame from each webcam.
    2.  Converts the frames to grayscale.
    3.  Detects keypoints and computes descriptors using SIFT.
    4.  Matches the descriptors using the Brute-Force matcher.
    5.  Sorts the matches by distance.
    6.  Draws the top 50 matches on a combined frame.
    7.  Calculates and displays the matching score and FPS.
    8.  Shows the final frame in a window.
    9.  Breaks the loop if the 'q' key is pressed.

### 6. Resource Cleanup
-   Once the loop is exited, `cam1.release()`, `cam2.release()`, and `cv2.destroyAllWindows()` are called to release the webcams and close all OpenCV windows.

## Video Demonstration

Link to video: [https://youtu.be/CmS2r868F3k](https://youtu.be/CmS2r868F3k)