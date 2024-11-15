import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Configuration
MODEL_PATH = './multi_class_detector.h5'
AIRPLANE_DIR = './data/aircraft'
CONTROLLER_DIR = './data/controllers'
CONFIDENCE_THRESHOLD = 0.85
MAX_PROPOSALS = 20
NMS_THRESHOLD = 0.6
SAMPLE_SIZE = 3

def load_images(directory, extensions=('.jpg', '.jpeg', '.png')):
    """Load image paths from a directory."""
    return [os.path.join(directory, img) for img in os.listdir(directory)
            if img.lower().endswith(extensions)]

def generate_proposals(image):
    """Generate region proposals using Selective Search."""
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return ss.process()[:MAX_PROPOSALS]

def apply_non_max_suppression(boxes, threshold):
    """Apply Non-Maximum Suppression to remove overlapping boxes."""
    if len(boxes) == 0:
        return []

    boxes = boxes.astype("float")
    pick = []

    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
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

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))

    return pick

def process_image(image_path, model):
    """Process a single image and return the annotated image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None

    proposals = generate_proposals(image)
    if len(proposals) == 0:
        print(f"No region proposals for image: {image_path}")
        return None

    boxes = []
    for x, y, w, h in proposals:
        region = image[y:y+h, x:x+w]
        resized = cv2.resize(region, (224, 224))
        prediction = model.predict(np.expand_dims(resized, axis=0))[0]
        
        label = "Controller" if np.argmax(prediction) == 1 else "Aircraft"
        score = prediction[1] if label == "Controller" else prediction[0]
        
        if score > CONFIDENCE_THRESHOLD:
            boxes.append([x, y, x+w, y+h, score, label])

    if len(boxes) == 0:
        print(f"No high-confidence predictions for image: {image_path}")
        return None

    boxes_array = np.array([box[:5] for box in boxes])
    labels = [box[5] for box in boxes]

    nms_indices = apply_non_max_suppression(boxes_array, NMS_THRESHOLD)
    nms_boxes = [np.append(boxes_array[i], labels[i]) for i in nms_indices if i < len(boxes_array)]

    for box in nms_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        score, label = float(box[4]), str(box[5])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    
    aircraft_images = load_images(AIRPLANE_DIR)
    controller_images = load_images(CONTROLLER_DIR)
    
    selected_images = shuffle(aircraft_images + controller_images)[:SAMPLE_SIZE * 2]
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, image_path in enumerate(selected_images):
        processed_image = process_image(image_path, model)
        if processed_image is not None:
            ax = axs[idx // 3, idx % 3]
            ax.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            ax.set_title(os.path.basename(image_path))
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()