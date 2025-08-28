# CS5330 Computer Vision Mini-Projects

This repository contains a collection of mini-projects developed for the CS5330 Computer Vision course. Each project explores different concepts and techniques in computer vision, from basic image manipulation to advanced topics like object detection and image segmentation.

## Mini-Projects

Here is a summary of the mini-projects included in this repository:

- ### [Mini-Project 3: Image World App](./mini-project3/)

  - **Description**: An application that applies various real-time video effects to a webcam feed. Users can toggle effects like translation, rotation, scaling, and perspective transformation using keyboard shortcuts.
  - **Key Technologies**: OpenCV, NumPy, Matplotlib.

- ### [Mini-Project 4: Dual Webcam Feature Matching](./mini-project4/)

  - **Description**: A real-time feature matching system that captures video from two webcams and uses the SIFT (Scale-Invariant Feature Transform) algorithm to detect and match features between the two feeds. It also displays a matching score and FPS.
  - **Key Technologies**: OpenCV, SIFT.

- ### [Mini-Project 5: Real-Time Panorama Creation](./mini-project5/)

  - **Description**: An application that creates a panoramic photo from a sequence of images captured from a live camera feed. It uses ORB (Oriented FAST and Rotated BRIEF) for feature detection and matching to stitch the images together.
  - **Key Technologies**: OpenCV, ORB.

- ### [Mini-Project 6: Lane Detection System](./mini-project6/)

  - **Description**: A vision-based lane detection system that processes video input to identify and highlight left and right lane lines on a road in real-time using the Hough Line Transform.
  - **Key Technologies**: OpenCV, Hough Line Transform.

- ### [Mini-Project 7: Live Classifier](./mini-project7/)

  - **Description**: A real-time object classification system that uses a Convolutional Neural Network (CNN) to classify objects in a live video feed into predefined classes (remote control, cell phone, TV, coffee mug).
  - **Key Technologies**: TensorFlow, Keras, OpenCV.

- ### [Mini-Project 9: Object Detection](./mini-project9/)

  - **Description**: This project is divided into three parts, exploring different object detection techniques:
    - **R-CNN for Multi-class Detection**: Implements a multi-class object recognition model using R-CNN to detect "remotes" and "airplanes". It also includes an optimized, lighter version of the model for better real-time performance.
    - **YOLO for Cups and Bananas**: Uses YOLO (You Only Look Once) to detect "cups" and "bananas" in real-time and automatically records a video clip when the target objects are detected.
    - **YOLOv5 for Stop Signs and Traffic Signals**: Implements a live detection system using YOLOv5 to recognize "Stop Signs" and "Traffic Lights" in driving scenarios.
  - **Key Technologies**: R-CNN, YOLO, YOLOv5, TensorFlow, OpenCV.

- ### [Mini-Project 10: U-Net for Vehicle Segmentation](./mini-project10/)

  - **Description**: Implements a U-Net architecture for vehicle segmentation. The project includes a complete pipeline for data collection, preprocessing, mask generation, model training, and optimization.
  - **Key Technologies**: U-Net, TensorFlow, Keras, OpenCV.

- ### [Mini-Project 11: Follow Everything](./mini-project11/)

  - **Description**: A real-time object detection and tracking system that combines YOLO for object detection and optical flow for tracking. The application can detect and follow multiple objects as they move in the video feed.
  - **Key Technologies**: YOLO, Optical Flow, OpenCV.
