import cv2 as cv
import numpy as np
import time

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

# Load the classes file
classesFile = "coco.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Set TARGET_CLASSES to detect only "remote" and "pottedplant"
TARGET_CLASSES = ["remote", "pottedplant"]

# Load YOLO configuration and weights
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

# Load the YOLO network
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Function to get the names of the output layers


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_out_layers, np.ndarray):
        return [layersNames[i - 1] for i in unconnected_out_layers.flatten()]
    else:
        raise RuntimeError("Failed to retrieve output layer names.")

# Function to draw bounding boxes


def drawPred(classId, conf, left, top, right, bottom, frame):
    label = f'{classes[classId]}: {conf:.2f}'
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    cv.putText(frame, label, (left, top - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Function to perform post-processing


def postprocess(frame, outs):
    frameHeight, frameWidth = frame.shape[:2]
    classIds, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x, center_y = int(
                    detection[0] * frameWidth), int(detection[1] * frameHeight)
                width, height = int(
                    detection[2] * frameWidth), int(detection[3] * frameHeight)
                left, top = int(center_x - width /
                                2), int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    if isinstance(indices, (list, tuple)):
        indices = [i[0] for i in indices]

    for i in indices:
        left, top, width, height = boxes[i]
        drawPred(classIds[i], confidences[i], left,
                 top, left + width, top + height, frame)

    return [classes[classIds[i]] for i in indices]


# Initialize webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize variables for saving video
frames = []
recording = False
start_time = None
save_fps = 5
video_counter = 1

while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print("Error: Could not read frame.")
        break

    # Prepare the frame for YOLO
    blob = cv.dnn.blobFromImage(
        frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))

    # Get detected classes
    detected_classes = postprocess(frame, outs)

    # Check if either "remote" or "pottedplant" is detected and start recording if not already recording
    if any(obj in TARGET_CLASSES for obj in detected_classes) and not recording:
        # Start recording and set the timer
        start_time = time.time()
        frames = []  # Reset frames for a new recording
        recording = True
        print(f"Started recording video #{video_counter}...")

    # Continue recording if in recording mode
    if recording:
        frames.append(frame)  # Store each frame in the list

        # Stop recording after 5 seconds
        if time.time() - start_time >= 5:
            recording = False

            # Define the output filename with an incrementing counter
            output_filename = f"CS5330_Group1_{video_counter}.mp4"
            video_counter += 1  # Increment the counter for the next video

            # Initialize the video writer with H.264 codec and lower FPS to slow down playback
            out = cv.VideoWriter(output_filename, cv.VideoWriter_fourcc(
                *'mp4v'), save_fps, (frame.shape[1], frame.shape[0]))

            # Write all frames to the output video file at the specified FPS
            for f in frames:
                out.write(f)
            out.release()
            print(f"Stopped recording and saved video as {output_filename}.")

    # Display the frame
    cv.imshow("Webcam YOLO Detection", frame)

    # Exit on 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
