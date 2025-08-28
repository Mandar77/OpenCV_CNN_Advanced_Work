# Mini-Project 6: Lane Detection System

## Overview

This project implements a computer vision-based lane detection system. It processes a video file of a road, identifies the lane lines, and overlays them on the video. The core of the system is the Hough Line Transform, a feature extraction technique for detecting lines in images. The script takes a video file as input, processes it frame by frame, and saves the output as a new video file.

## Features

- **Video Processing**: Processes video files to perform lane detection.
- **Lane Line Identification**: Identifies both the left and right lane lines on the road.
- **Hough Line Transform**: Uses the Hough Line Transform to detect lines from edge-detected images.
- **Line Averaging**: Averages the detected line segments to create a single, stable line for each lane.
- **Visual Overlay**: Draws the detected lanes as red lines on the output video.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)

## Installation

1.  Ensure Python 3.x is installed.
2.  Install the required libraries:
    ```bash
    pip install opencv-python numpy
    ```

## Usage

Run the script from the command line, providing the path to the input video file and a name for the output video file.

```bash
python mini-project6.py -f <input_video.mp4> -o <output_video.mp4>
```

For example:
```bash
python mini-project6.py -f lane_test1.mp4 -o output1.mp4
```

-   `-f` or `--file`: Path to the input video file.
-   `-o` or `--out`: Name of the output video file.

## Lane Detection Pipeline

The lane detection process is a pipeline of several steps that are applied to each frame of the video:

1.  **Preprocessing (`preprocess`)**:
    -   The frame is converted to grayscale.
    -   A Gaussian blur is applied to reduce noise and smooth the image.
    -   The Canny edge detector is used to find the edges in the image.

2.  **Region of Interest (`region_of_interest`)**:
    -   A triangular mask is created to define a region of interest (ROI). This focuses the detection on the area of the image where the lanes are most likely to be.
    -   The edge-detected image is masked with this ROI.

3.  **Hough Line Transform (`hough_lines`)**:
    -   The `cv2.HoughLinesP` function is applied to the masked image. This detects straight line segments in the ROI.

4.  **Line Averaging (`average_slope_intercept`)**:
    -   The lines detected by the Hough Transform are often fragmented. This step averages them to produce a single, continuous line for each lane.
    -   Lines are separated into left and right lanes based on their slope (negative for left, positive for right).
    -   The slope and intercept of the lines in each group are averaged.

5.  **Line Creation (`create_lines`)**:
    -   The averaged slope and intercept are used to calculate the start and end points of the final lane lines, extending them to a fixed vertical range.

6.  **Drawing the Lines (`draw_lane_lines`)**:
    -   The final, averaged lane lines are drawn in red on a blank image.
    -   This image is then overlaid on the original frame to produce the final output.

## Code Structure

-   **`preprocess(frame)`**: Performs grayscale conversion, blurring, and Canny edge detection.
-   **`region_of_interest(edges)`**: Masks the image to a specific region.
-   **`hough_lines(edges)`**: Applies the Hough Line Transform.
-   **`average_slope_intercept(lines)`**: Averages the detected lines.
-   **`create_lines(y1, y2, line)`**: Creates line coordinates from slope and intercept.
-   **`draw_lane_lines(frame, lines)`**: Draws the final lane lines on the frame.
-   **`process_video(input_file, output_file)`**: The main function that reads the video, processes each frame through the pipeline, and saves the output.
-   **`if __name__ == '__main__':`**: Parses command-line arguments and calls `process_video`.

## Limitations

-   The system may struggle with sharp curves or turns, as it is designed for relatively straight roads.
-   Performance can be affected by changing light conditions, shadows, and the clarity of the lane markings.
-   The region of interest is a fixed triangle, which may not be suitable for all road perspectives.
