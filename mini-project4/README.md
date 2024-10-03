# Dual Webcam Feature Matching

## Overview

This Python script demonstrates real-time feature matching between two webcam feeds using OpenCV. It captures video from two separate webcams, detects and matches features between the frames, and displays the results in real-time. The script also calculates and displays a matching score and the frames per second (FPS) of the processing.

## Features

- **Dual Webcam Support**: Simultaneously captures video from two webcams.
- **Feature Detection**: Utilizes SIFT (Scale-Invariant Feature Transform) for robust feature detection.
- **Real-time Matching**: Performs feature matching between frames from both webcams in real-time.
- **Visual Feedback**: Displays matched features visually on a combined frame.
- **Performance Metrics**: 
  - Calculates and displays a matching score based on feature distances.
  - Shows real-time FPS to indicate processing speed.

## Requirements

- Python 3.x
- OpenCV (cv2)
- OpenCV-contrib-python (for SIFT functionality)
- Two webcams connected to the computer

## Installation

1. Ensure Python 3.x is installed on your system.
2. Install the required libraries:


## Usage

1. Connect two webcams to your computer.
2. Run the script
3. The script will open a window showing the combined feed from both webcams with matched features.
4. Press 'q' to exit the program.

## Code Structure

- **Webcam Initialization**: Sets up two VideoCapture objects for the webcams.
- **Feature Detector**: Initializes SIFT for feature detection.
- **Matcher Setup**: Configures a Brute-Force matcher for feature matching.
- **Main Loop**:
- Captures frames from both webcams.
- Detects features using SIFT.
- Matches features between the two frames.
- Draws matched features and displays performance metrics.
- Updates the display in real-time.

## Video Demonstration of the Application:

Link to video: https://youtu.be/CmS2r868F3k