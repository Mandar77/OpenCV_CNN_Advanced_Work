import os
import warnings
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

# Configuration
REMOTE_DATA_DIR = "remotes/"
AIRPLANE_DATA_DIR = "airplanes/"
REMOTE_CSV = "remotes_annotations.csv"
AIRPLANE_CSV = "airplanes_annotations.csv"
MAX_PROPOSALS = 100
POSITIVE_IOU_THRESHOLD = 0.5
NEGATIVE_IOU_THRESHOLD = 0.1
SAMPLE_SIZE = 30
EPOCHS = 5
BATCH_SIZE = 32

def compute_iou(box1, box2):
    """Compute the Intersection over Union (IoU) of two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def generate_proposals(image):
    """Generate region proposals using a sliding window approach."""
    height, width = image.shape[:2]
    proposals = []
    
    window_sizes = [(64, 64), (128, 128), (256, 256)]
    step_size = 32
    
    for (w, h) in window_sizes:
        for y in range(0, height - h, step_size):
            for x in range(0, width - w, step_size):
                proposals.append((x, y, x + w, y + h))
    
    return proposals[:MAX_PROPOSALS]

def process_image(image_path, annotation, class_label):
    """Process a single image and its annotations."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return [], []

    ground_truth = [int(annotation['x_min']), int(annotation['y_min']),
                    int(annotation['x_max']), int(annotation['y_max'])]

    proposals = generate_proposals(image)
    
    train_data = []
    svm_data = []

    gt_region = image[ground_truth[1]:ground_truth[3], ground_truth[0]:ground_truth[2]]
    gt_resized = cv2.resize(gt_region, (224, 224))
    svm_data.append((gt_resized, class_label))

    positive_count, negative_count = 0, 0
    for proposal in proposals:
        if positive_count >= SAMPLE_SIZE and negative_count >= SAMPLE_SIZE:
            break

        iou = compute_iou(ground_truth, proposal)
        
        if iou > POSITIVE_IOU_THRESHOLD and positive_count < SAMPLE_SIZE:
            label = 1
            positive_count += 1
        elif iou < NEGATIVE_IOU_THRESHOLD and negative_count < SAMPLE_SIZE:
            label = 0
            negative_count += 1
        else:
            continue

        region = image[proposal[1]:proposal[3], proposal[0]:proposal[2]]
        resized = cv2.resize(region, (224, 224))
        train_data.append((resized, label))

        if negative_count < 5 and label == 0:
            svm_data.append((resized, [1 - class_label[1], class_label[1]]))

    return train_data, svm_data

def create_datasets():
    """Create datasets for training."""
    for csv_file, image_dir, class_label in [
        (REMOTE_CSV, REMOTE_DATA_DIR, [0, 1]),
        (AIRPLANE_CSV, AIRPLANE_DATA_DIR, [1, 0])
    ]:
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            image_path = os.path.join(image_dir, row['image_path'])
            if os.path.exists(image_path):
                train_data, svm_data = process_image(image_path, row, class_label)
                for img, label in train_data:
                    yield (img, label), 'train'
                for img, label in svm_data:
                    yield (img, label), 'svm'

def data_generator(batch_size=32):
    train_batch, svm_batch = [], []
    for (img, label), data_type in create_datasets():
        if data_type == 'train':
            train_batch.append((img, label))
            if len(train_batch) == batch_size:
                yield np.array([x[0] for x in train_batch]), np.array([x[1] for x in train_batch]), None, None
                train_batch = []
        else:  # svm
            svm_batch.append((img, label))
            if len(svm_batch) == batch_size:
                yield None, None, np.array([x[0] for x in svm_batch]), np.array([x[1] for x in svm_batch])
                svm_batch = []
    
    # Yield any remaining data
    if train_batch:
        yield np.array([x[0] for x in train_batch]), np.array([x[1] for x in train_batch]), None, None
    if svm_batch:
        yield None, None, np.array([x[0] for x in svm_batch]), np.array([x[1] for x in svm_batch])

def create_base_model():
    """Create a base model using VGG16."""
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(base_model.input, output)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_final_model(base_model):
    """Create the final model for SVM-like classification."""
    x = base_model.layers[-2].output
    output = Dense(2)(x)
    model = Model(base_model.input, output)
    model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
    return model

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

    return boxes[pick].astype("int")

def main():
    # Create models
    base_model = create_base_model()
    final_model = create_final_model(base_model)

    # Train the models
    gen = data_generator(BATCH_SIZE)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for batch, (train_images, train_labels, svm_images, svm_labels) in enumerate(gen):
            if train_images is not None:
                loss = base_model.train_on_batch(train_images, train_labels)
                print(f"Batch {batch+1} - Base model loss: {loss}")
            if svm_images is not None:
                loss = final_model.train_on_batch(svm_images, svm_labels)
                print(f"Batch {batch+1} - Final model loss: {loss}")

    # Save models
    base_model.save('base_model.h5')
    final_model.save('multi_class_detector_light.h5')

    # Test on a sample image
    test_image_path = os.path.join(REMOTE_DATA_DIR, 'remote20.jpg')
    test_image = cv2.imread(test_image_path)
    if test_image is not None:
        proposals = generate_proposals(test_image)

        detection_boxes = []
        for x1, y1, x2, y2 in proposals[:50]:
            region = test_image[y1:y2, x1:x2]
            resized = cv2.resize(region, (224, 224))
            prediction = final_model.predict(np.expand_dims(resized, axis=0))[0]
            if prediction[1] > 0.5:
                detection_boxes.append([x1, y1, x2, y2, prediction[1]])

        nms_boxes = apply_non_max_suppression(np.array(detection_boxes), 0.3)

        for box in nms_boxes:
            x1, y1, x2, y2 = box[:4]
            cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite('test_result.jpg', test_image)
        print("Test result saved as 'test_result.jpg'")
    else:
        print("Test image not found.")

if __name__ == "__main__":
    main()