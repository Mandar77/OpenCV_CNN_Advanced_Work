# Mini-Project 5: Real-Time Panorama Creation

## Overview

This project is a real-time application that creates a panoramic image from a sequence of frames captured from a live webcam feed. The application uses OpenCV for video capture and feature-based image stitching. It employs ORB (Oriented FAST and Rotated BRIEF) for feature detection and a Brute-Force matcher to find corresponding features between images. The captured frames are then stitched together using a homography transformation to create the final panorama.

## Features

- **Live Panorama Creation**: Stitch images from a live video feed to create a panorama.
- **ORB Feature Detection**: Uses the ORB algorithm for efficient and robust feature detection.
- **Multithreaded Video Capture**: A separate thread is used for video capture to prevent lag in the main application.
- **Selective Frame Capture**: Only captures new frames if they are significantly different from the previous one, avoiding duplicates.
- **Interactive Controls**: Start and stop frame capture using keyboard shortcuts.

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

## How to Run

1.  Make sure you have a webcam connected to your computer.
2.  Run the script from your terminal:
    ```bash
    python mini-project.py
    ```
3.  A window will open showing the live video feed. Follow the controls below to create a panorama.
4.  Press 'q' to quit the application.

## Controls

-   **`s`**: **Start** capturing frames. Move your camera slowly to capture the scene you want in your panorama.
-   **`a`**: **Stop** capturing frames and **create** the panorama. The application will stitch the captured frames and display the result. The final panorama is saved as `panorama.jpg`.
-   **`q`**: **Quit** the application.

## Implementation Details

### 1. Multithreaded Video Capture (`VideoCaptureThread`)
To ensure the main thread is not blocked by I/O operations from the webcam, video capturing is handled in a separate thread. The `VideoCaptureThread` class continuously reads frames from the webcam in the background.

### 2. Feature Detection and Matching
-   **`orb_feature_detection()`**: This function uses `cv2.ORB_create()` to detect keypoints and compute their descriptors. ORB is a fast and efficient alternative to SIFT and SURF.
-   **`match_features()`**: This function uses a `cv2.BFMatcher` (Brute-Force Matcher) with `cv2.NORM_HAMMING` distance, which is suitable for binary descriptors like the ones produced by ORB. It finds the best matches between the descriptors of two images.

### 3. Image Stitching (`stitch_images`)
This is the core function for stitching two images together:
1.  It first finds features in both images using `orb_feature_detection`.
2.  It then matches these features using `match_features`.
3.  If enough good matches are found (more than 10), it uses `cv2.findHomography` to compute the perspective transformation matrix (homography) between the two images. The `RANSAC` method is used to make the estimation robust against outliers.
4.  Finally, it uses `cv2.warpPerspective` to warp one image to align with the other, creating a stitched image.

### 4. Panorama Creation (`create_panorama`)
This function takes a list of captured frames and iteratively stitches them together. It starts with the first frame and progressively stitches the next frame onto the current panorama.

### 5. Selective Frame Capture (`is_frame_different`)
To avoid capturing many identical or very similar frames (which is common when the camera is not moving), this function checks if a new frame is significantly different from the last captured frame. It does this by calculating the absolute difference between the two frames and counting the number of non-zero pixels. A new frame is only added if this count exceeds a certain threshold.

### 6. Main Function (`main`)
The `main` function orchestrates the application:
-   It initializes the `VideoCaptureThread`.
-   It enters a loop that displays the live video feed.
-   It listens for keyboard input to start or stop capturing frames.
-   When capturing is active, it calls `is_frame_different` to decide whether to add the current frame to a list.
-   When the user stops capturing, it calls `create_panorama` to generate and display the final image.

## Video Demonstration

Link: [https://youtu.be/GoV3AqAtraQ](https://youtu.be/GoV3AqAtraQ)