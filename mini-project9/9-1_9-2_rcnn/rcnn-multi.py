# new version
import os
import cv2
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras import Model
# import ssl
# import certifi

# ssl._create_default_https_context = ssl.create_default_context
# ssl._create_default_https_context().load_verify_locations(certifi.where())

# Paths to datasets and labels
# Folder with remote images (JPG, PNG, etc.)
remote_image_path = "./data/remotes"
remote_label_dir = "./data/images_csv"     # Folder with CSV files for remotes

airplane_image_path = "./data/sm_images"   # Folder with airplane images
# Folder with CSV files for airplanes
airplane_label_dir = "./data/sm_annotations"


# Lists to store data
train_images = []
train_labels = []
svm_images = []
svm_labels = []

# IoU calculation


def get_iou(bb1, bb2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return max(0.0, min(1.0, iou))


# Step 1: Running Selective Search on individual images to obtain region proposals (2000 here).
# Enable optimized computation in OpenCV
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

for e, i in enumerate(os.listdir(remote_label_dir) + os.listdir(airplane_label_dir)):
    try:
        # Determine class based on the file prefix and set paths accordingly
        if i.startswith("remote"):
            class_label = [0, 1]  # Remote class
            base_filename = i.split(".")[0]
            image_path = remote_image_path
            label_dir = remote_label_dir
        elif i.startswith("airplane"):
            class_label = [1, 0]  # Airplane class
            base_filename = i.split(".")[0]
            image_path = airplane_image_path
            label_dir = airplane_label_dir
        else:
            continue  # Skip files that do not match either class

        # Check for image files with .jpg, .jpeg, and .png extensions
        jpg_filename = base_filename + ".jpg"
        jpeg_filename = base_filename + ".jpeg"
        png_filename = base_filename + ".png"

        # Load image by checking file extensions in order
        if os.path.exists(os.path.join(image_path, jpg_filename)):
            filename = jpg_filename
        elif os.path.exists(os.path.join(image_path, jpeg_filename)):
            filename = jpeg_filename
        elif os.path.exists(os.path.join(image_path, png_filename)):
            filename = png_filename
        else:
            print(f"No image found for {base_filename}")
            continue

        print(e, filename)
        image = cv2.imread(os.path.join(image_path, filename))
        if image is None:
            print(f"Error loading image: {filename}")
            continue

        # Load ground truth bounding boxes from CSV
        df = pd.read_csv(os.path.join(label_dir, i))
        gtvalues = []

        for _, row in df.iterrows():
            x1, y1, x2, y2 = map(int, row.iloc[0].split(" "))
            gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})

            # Crop ground truth images for SVM training
            timage = image[y1:y2, x1:x2]
            resized = cv2.resize(timage, (224, 224),
                                 interpolation=cv2.INTER_AREA)
            svm_images.append(resized)
            svm_labels.append(class_label)  # Assign appropriate class label

        # Set up Selective Search and process image for region proposals
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        imout = image.copy()
        counter, falsecounter = 0, 0
        flag = False

        # Step 2: Classify region proposals as positive and negative examples based on IoU
        for e, result in enumerate(ssresults):
            if e < 2000 and not flag:
                x, y, w, h = result
                for gtval in gtvalues:
                    iou = get_iou(
                        gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                    timage = imout[y:y + h, x:x + w]

                    if counter < 30 and iou > 0.7:  # Positive examples for training
                        resized = cv2.resize(
                            timage, (224, 224), interpolation=cv2.INTER_AREA)
                        train_images.append(resized)
                        train_labels.append(1)  # Positive label
                        counter += 1
                    elif falsecounter < 30 and iou < 0.3:  # Negative examples for training
                        resized = cv2.resize(
                            timage, (224, 224), interpolation=cv2.INTER_AREA)
                        train_images.append(resized)
                        train_labels.append(0)  # Negative label
                        falsecounter += 1
                    if falsecounter < 5 and iou < 0.3:  # Negative examples for SVM
                        resized = cv2.resize(
                            timage, (224, 224), interpolation=cv2.INTER_AREA)
                        svm_images.append(resized)
                        # Inverted label
                        svm_labels.append([1 - class_label[1], class_label[1]])

                if counter >= 30 and falsecounter >= 30:
                    flag = True  # Stop if we have enough examples

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue


# Conversion of train data into arrays for further training
X_new = np.array(train_images)
Y_new = np.array(train_labels)

# Step 3: Passing every proposal through a pretrained network (VGG16 trained on ImageNet) to output a fixed-size feature vector (4096 here).
vgg = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')
for layer in vgg.layers[:-2]:
    layer.trainable = False
x = vgg.get_layer('fc2').output
x = Dense(1, activation='sigmoid')(x)
model = Model(vgg.input, x)
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc'])
model.summary()
# model.fit(X_new, Y_new, batch_size=16, epochs=1,
#           verbose=1, validation_split=0.05, shuffle=True)
model.fit(X_new, Y_new, batch_size=16, epochs=5,
          verbose=1, validation_split=0.05, shuffle=True)


# Step 4: Using this feature vector to train an SVM.
x = model.get_layer('fc2').output
Y = Dense(2)(x)
final_model = Model(model.input, Y)
final_model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
final_model.summary()

# Train SVM model
# hist_final = final_model.fit(np.array(svm_images), np.array(svm_labels),
#                              batch_size=16, epochs=2, verbose=1,
#                              shuffle=True, validation_split=0.05)
hist_final = final_model.fit(np.array(svm_images), np.array(svm_labels),
                             batch_size=16, epochs=5, verbose=1,
                             shuffle=True, validation_split=0.05)

final_model.save('my_model_weights.h5')

# Step 5: Non-maximum Suppression (NMS) to remove redundant overlapping bounding boxes


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
    return boxes[pick].astype("int")


# Plotting loss and analyzing losses
plt.plot(hist_final.history['loss'])
plt.plot(hist_final.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss", "Validation Loss"])
plt.savefig('chart_loss.png')
plt.show()

# Testing on a new image (update the filename as needed)
test_image_filename = "remote20.jpg"
image = cv2.imread(os.path.join(image_path, test_image_filename))
if image is not None:
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()
    imOut = image.copy()
    boxes = []

    for e, result in enumerate(ssresults):
        if e < 50:
            x, y, w, h = result
            timage = image[y:y + h, x:x + w]
            resized = cv2.resize(timage, (224, 224),
                                 interpolation=cv2.INTER_AREA)
            resized = np.expand_dims(resized, axis=0)
            out = final_model.predict(resized)
            score = out[0][1]
            if score > 0.5:
                boxes.append([x, y, x + w, y + h, score])

    boxes = np.array(boxes)
    nms_boxes = non_max_suppression(boxes, overlapThresh=0.3)
    for box in nms_boxes:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(imOut, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

    plt.imshow(imOut)
    plt.show()
else:
    print(f"Test image {test_image_filename} not found.")
