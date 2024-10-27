# USAGE: python LiveClassifier.py -f video_file_name -o out_video.avi -m model_path

import cv2
import numpy as np
import time
import os
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name")
parser.add_argument("-m", "--model", type=str, required=True, help="Path to trained model")

args = parser.parse_args()

# Load the trained model
model = load_model(args.model)

# Define class labels
class_labels = ['Remote Control', 'Cell Phone', 'TV', 'Coffee Mug']

# Check if the file argument is provided, otherwise use the camera
if args.file:
    vs = cv2.VideoCapture(args.file)
else:
    vs = cv2.VideoCapture(0)  # 0 is the default camera

time.sleep(2.0)

# Get the default resolutions
width  = int(vs.get(3))
height = int(vs.get(4))

# Define the codec and create a VideoWriter object
out_filename = args.out if args.out else 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height), True)

def preprocess_image(image):
    # Resize and preprocess the image for the model
    image = cv2.resize(image, (224, 224))  # Adjust size based on your model's input
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize pixel values
    return image

# Initialize variables for FPS calculation
fps_start_time = time.time()
fps = 0
frame_count = 0

# loop over the frames from the video stream
while True:
    # grab the frame from video stream
    ret, frame = vs.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_image(frame)

    # Make prediction
    prediction = model.predict(processed_frame)
    class_index = np.argmax(prediction[0])
    confidence = prediction[0][class_index] * 100

    # Get the predicted class label
    predicted_class = class_labels[class_index]

    # Draw the label and confidence on the frame
    label = f"{predicted_class}: {confidence:.2f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Calculate and display FPS
    frame_count += 1
    if (time.time() - fps_start_time) > 1:
        fps = frame_count / (time.time() - fps_start_time)
        fps_start_time = time.time()
        frame_count = 0

    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Write the frame to the output video file
    out.write(frame)

    # show the output frame
    cv2.imshow("Live Classifier", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Release the video capture object
vs.release()
out.release()
cv2.destroyAllWindows()