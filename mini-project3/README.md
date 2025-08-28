# Mini-Project 3: Image World App - Real-Time Video Effects

## By Group 1

This project is a real-time video processing application that applies various geometric transformations to a live webcam feed using OpenCV. The user can toggle different effects on and off using keyboard shortcuts. The application displays both the original and the modified video streams side-by-side for easy comparison.

## Features

- **Live Video Processing**: Captures video from a webcam and processes it in real-time.
- **Multiple Video Effects**: Supports several geometric transformations:
    - Translation
    - Rotation
    - Scaling
    - Perspective Transformation
- **Interactive Controls**: Use keyboard shortcuts to toggle effects on and off.
- **Side-by-Side Display**: Shows the original video feed next to the transformed feed.
- **FPS Counter**: Displays the current frames per second (FPS) of the video stream.

## Project Set-up

### Prerequisites
- Python 3.x
- pip

### Packages to install
```bash
pip install opencv-python numpy matplotlib
```

## How to Run

1.  Make sure you have a webcam connected to your computer.
2.  Run the script from your terminal:
    ```bash
    python WebCam2.py
    ```
3.  A window will open showing the live video. Press the keys listed below to apply effects.
4.  Press 'q' to quit the application.

## Controls

You can press the following keys on your keyboard to toggle different video effects. Pressing a key once activates the effect, and pressing it again deactivates it.

-   **`t`**: **Translate** the video frame from (0,0) to (50,50).
-   **`r`**: **Rotate** the video frame by 45 degrees around the top-left corner.
-   **`s`**: **Scale** the image to 1.5 times its original size.
-   **`p`**: Apply a **Perspective Transformation**.
-   **`q`**: **Quit** the application.

## Implementation Details

The application is built around a main loop that continuously captures frames from the webcam and applies transformations based on user input.

### Code Structure

-   **`modes` dictionary**: A dictionary that holds the state ( `True` or `False`) for each transformation mode. This allows the application to track which effect is currently active.

-   **`reset_modes()`**: A helper function to set all transformation modes to `False`. This is called whenever a new mode is activated to ensure that only one effect is active at a time.

-   **`handle_key_press(key)`**: This function maps keyboard inputs to their corresponding modes. When a key is pressed, it toggles the state of the corresponding mode in the `modes` dictionary.

-   **`calculate_and_display_fps(...)`**: This function calculates the frames per second (FPS) and overlays it on the video frame.

-   **Main Loop**:
    1.  **Capture Frame**: Reads a frame from the webcam.
    2.  **Apply Transformation**: Checks the `modes` dictionary to see if any effect is active. If so, it applies the corresponding transformation matrix (`M`) to the frame using OpenCV's `warpAffine` or `warpPerspective` functions.
    3.  **Display Frames**: The original frame and the modified frame are concatenated horizontally using `cv2.hconcat`. This combined frame is then displayed in a window.
    4.  **Listen for Keystrokes**: `cv2.waitKey(1)` waits for a key press. The input is then passed to `handle_key_press` to update the active mode. The loop terminates if 'q' is pressed.

### Transformations

-   **Translation**: Shifts the image by a specified amount in the x and y directions.
-   **Rotation**: Rotates the image around a specified center point by a given angle.
-   **Scaling**: Resizes the image by a certain factor.
-   **Perspective Transformation**: Distorts the perspective of the image, which can be used to correct for perspective or create a 3D effect. The transformation is defined by a source and destination set of four points.

## Video Demonstration

Link to video: [https://youtu.be/zMZbE_ToIWE](https://youtu.be/zMZbE_ToIWE)
