# import the necessary packages
import cv2
import numpy as np
import time
import os
import argparse
import tensorflow as tf

# Load the pre-trained model
final_model = tf.keras.models.load_model('my_model_weights.h5')

# Non-maximum Suppression function


def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > overlapThresh)[0])))
    return pick  # Return the indices instead of boxes


# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name")

args = parser.parse_args()

# Check if the file argument is provided, otherwise use the camera
if args.file:
    vs = cv2.VideoCapture(args.file)
else:
    vs = cv2.VideoCapture(0)  # 0 is the default camera

time.sleep(2.0)

# Get the default resolutions
width = int(vs.get(3))
height = int(vs.get(4))

# Define the codec and create a VideoWriter object
out_filename = args.out
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height), True)

# Initialize frame counters for FPS calculation
fps_start_time = time.time()
frame_count = 0
frame_skip = 2  # Process every second frame

# loop over the frames from the video stream
while True:
    # grab the frame from video stream
    ret, frame = vs.read()
    if not ret:
        break

    # Skip frames to improve FPS
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    # Increment the frame count
    frame_count += 1
    fps_end_time = time.time()
    fps = frame_count / (fps_end_time - fps_start_time)

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    # Apply Selective Search to propose regions
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(small_frame)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()

    imOut = frame.copy()
    boxes = []
    labels = []  # Separate list for labels

    # Process each region proposal
    # Limit proposals for performance
    for e, result in enumerate(ssresults[:15]):
        x, y, w, h = result
        # Scale bounding box to original size
        x, y, w, h = x * 2, y * 2, w * 2, h * 2
        timage = frame[y:y + h, x:x + w]
        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
        resized = np.expand_dims(resized, axis=0)
        out = final_model.predict(resized)
        label = "Remote" if np.argmax(out[0]) == 1 else "Airplane"
        score = out[0][1] if label == "Remote" else out[0][0]

        if score > 0.85:
            # Only numerical data here
            boxes.append([x, y, x + w, y + h, score])
            labels.append(label)  # Keep label in a separate list

    # Convert list of boxes to numpy array for NMS
    boxes = np.array(boxes, dtype=object)

    # Apply Non-maximum Suppression (NMS)
    nms_indices = non_max_suppression(boxes, overlapThresh=0.5)

    # Keep only boxes and labels from the selected NMS indices
    nms_boxes = [np.append(boxes[i], labels[i]) for i in nms_indices]

    # Draw bounding boxes and label on the frame
    for box in nms_boxes:
        x1, y1, x2, y2, score, label = box
        # Ensure coordinates are integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(imOut, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(imOut, f"{label} ({float(score):.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Display FPS on the frame
    cv2.putText(imOut, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Write the frame to the output video file
    if args.out:
        out.write(imOut)

    # show the output frame
    cv2.imshow("Frame", imOut)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Release the video capture object
vs.release()
out.release()
cv2.destroyAllWindows()
