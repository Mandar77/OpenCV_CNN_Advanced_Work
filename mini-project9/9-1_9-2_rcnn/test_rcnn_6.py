import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# Load the pre-trained model
final_model = tf.keras.models.load_model('./my_model_weights.h5')

# Non-maximum Suppression function


def non_max_suppression(boxes, overlapThresh):
    """Perform non-maximum suppression on bounding boxes."""
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

    return pick  # Return indices instead of boxes


# Directories for images
plane_image_dir = './data/sm_images'  # Plane images directory
remote_image_dir = './data/remotes'   # Remote images directory

# Get a list of all images in each directory with .jpg, .jpeg, or .png extensions
plane_images = [os.path.join(plane_image_dir, img) for img in os.listdir(
    plane_image_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
remote_images = [os.path.join(remote_image_dir, img) for img in os.listdir(
    remote_image_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Randomly select 3 images from each directory
selected_plane_images = random.sample(plane_images, 3)
selected_remote_images = random.sample(remote_images, 3)

# Combine the selected images
selected_images = selected_plane_images + selected_remote_images

# Shuffle to mix airplane and remote images
random.shuffle(selected_images)

# Create a figure for plotting
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Process and display each selected image
for idx, image_path in enumerate(selected_images):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        continue

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()

    if ssresults is None or len(ssresults) == 0:
        print(f"No region proposals for image: {image_path}")
        continue

    imOut = image.copy()
    boxes = []

    # Process each region proposal and make a prediction
    # Limit to the top 50 region proposals for performance
    for e, result in enumerate(ssresults[:20]):
        x, y, w, h = result
        timage = image[y:y + h, x:x + w]
        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
        resized = np.expand_dims(resized, axis=0)
        out = final_model.predict(resized)

        # Determine label based on the model's prediction
        label = "Remote" if np.argmax(out[0]) == 1 else "Airplane"

        # Collect bounding boxes and corresponding confidence scores
        score = out[0][1] if label == "Remote" else out[0][0]
        if score > 0.85:  # Threshold to filter low confidence detections
            # Append label with box
            boxes.append([x, y, x + w, y + h, score, label])

    # Ensure that boxes array is populated correctly
    if len(boxes) == 0:
        print(f"No high-confidence predictions for image: {image_path}")
        continue

    # Convert list of boxes to numpy array for NMS (exclude labels for now)
    boxes_array = np.array([box[:5] for box in boxes],
                           dtype="float")  # Only numerical data
    labels = [box[5] for box in boxes]  # Extract labels separately

    # Apply Non-maximum Suppression (NMS)
    nms_indices = non_max_suppression(boxes_array, overlapThresh=0.6)

    # Ensure indices are within bounds before re-attaching labels
    nms_boxes = [np.append(boxes_array[i], labels[i])
                 for i in nms_indices if i < len(boxes_array)]

    # Draw bounding boxes and label on the image
    # Draw bounding boxes and label on the image
    for box in nms_boxes:
        # Cast the bounding box coordinates to float and then to int for drawing
        x1, y1, x2, y2 = map(lambda v: int(float(v)), box[:4])
        score = float(box[4])  # Ensure score is a float
        label = str(box[5])    # Ensure label is a string

        # Draw the bounding box and label on the image
        cv2.rectangle(imOut, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(imOut, f"{label} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Display the image with bounding boxes
    ax = axs[idx // 3, idx % 3]
    ax.imshow(cv2.cvtColor(imOut, cv2.COLOR_BGR2RGB))
    ax.set_title(os.path.basename(image_path))
    ax.axis('off')

plt.tight_layout()
plt.show()
