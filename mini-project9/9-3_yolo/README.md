# CS5330 Group 1 - Project 9-3 README: YOLO Object Detection for Cups and Bananas
## Step 1: Setting Up YOLO with WebCamSave_Yolo.py
We developed a custom script WebCamSave_Yolo.py to implement YOLO (You Only Look Once) object detection with a focus on identifying cups and bananas. This script utilizes a deep neural network for real-time object recognition and automatic video recording.
- Script Setup:
  1. The script initializes the YOLO model using a configuration file (yolov3.cfg) and pre-trained weights (yolov3.weights).
  2. It loads coco.names, which contains the list of objects the model can detect.
  3. We specifically set the target classes to "cup" and "banana".
- Key Components:
  1. ObjectDetector Class: Handles YOLO model initialization and object detection.
  2. VideoRecorder Class: Manages automatic video recording when target objects are detected.
- Step 2: Running the Object Detection System
To run the system:
1. Ensure all required files (yolov3.cfg, yolov3.weights, coco.names) are in the same directory as the script.
1. Execute the script:

`python WebCamSave_Yolo.py`

## Process:
- The script initializes the webcam and begins real-time object detection.
- When a cup or banana is detected, it automatically starts recording a 5-second video clip.

## Output:

- The script displays a live feed with detected objects marked by bounding boxes and labels.
- Video clips (e.g., DetectedObject_1.mp4, DetectedObject_2.mp4) are saved when cups or bananas are detected.

### Key Features and Code Structure
1. Target Class Selection:
- We set TARGET_CLASSES = ["cup", "banana"] to focus detection on these specific objects.

2. ObjectDetector Class:
- detect_objects(frame): Processes each frame for object detection.
- process_detections(frame, outputs): Filters detections and draws bounding boxes.

3. VideoRecorder Class:
- start_recording(): Initiates video recording when a target object is detected.
- stop_recording(): Saves the recorded video clip after 5 seconds.

4. Main Execution Flow:
- Continuous frame capture from the webcam.
- Object detection on each frame.
- Automatic video recording triggered by cup or banana detection.

5. Customization Options
- Modifying Target Objects: Change the TARGET_CLASSES list in the main() function to detect different objects.
- Adjusting Detection Sensitivity: Modify conf_threshold and nms_threshold in the ObjectDetector initialization.
- Changing Recording Duration: Alter the time check in VideoRecorder.should_stop() method.

6. Video Saving Parameters: 
- Adjust save_fps in VideoRecorder initialization to change the frame rate of saved videos.

### How to Run
1. Ensure yolov3.weights, yolov3.cfg, and coco.names are in the script's directory.

2. Run the script:

`python WebCamSave_Yolo.py`

3. The system will start detecting cups and bananas, automatically recording video clips when these objects appear.

4. Press 'q' to exit the program.


This implementation demonstrates the application of YOLO object detection for specific object recognition and automated video capture, showcasing its potential in various real-world scenarios.