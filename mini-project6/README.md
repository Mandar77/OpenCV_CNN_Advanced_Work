# Lane Detection System

This project implements a vision-based lane detection system using the Hough Line Transform. It processes video input to identify and highlight left and right lane lines on a road in real-time.

## Features

- Processes video files or camera input
- Identifies left and right lane lines
- Draws detected lanes as red lines on the output video
- Real-time processing and display

## Requirements

- Python 3.x
- OpenCV
- NumPy


## Usage

Run the script from the command line, providing input and output file names:

- `-f` or `--file`: Path to the input video file
- `-o` or `--out`: Name of the output video file

## How it works

1. Preprocesses each frame (grayscale conversion, Gaussian blur, Canny edge detection)
2. Defines a region of interest
3. Applies Hough Line Transform to detect lines
4. Separates and averages left and right lanes
5. Draws detected lanes on the original frame

## Limitations

- May not adapt quickly to sharp curves or turns
- Performance can vary based on lighting conditions and lane clarity
