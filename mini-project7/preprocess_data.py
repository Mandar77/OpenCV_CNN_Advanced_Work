import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

def preprocess_dataset(dataset_path, target_size=(224, 224)):
    X = []
    y = []
    class_labels = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for i, class_name in enumerate(class_labels):
        class_path = os.path.join(dataset_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            try:
                # Use PIL to open the image, which supports various formats
                with Image.open(image_path) as img:
                    # Convert to RGB if it's not already
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Resize the image
                    img = img.resize(target_size)
                    # Convert to numpy array
                    image = np.array(img)
                    # Convert to float and normalize
                    image = image.astype(np.float32) / 255.0
                    X.append(image)
                    y.append(i)
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
    
    if not X:
        raise ValueError("No valid images found in the dataset")
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, class_labels

# Preprocess the dataset
dataset_path = 'D:/KhouryGithub/CS5330_FA24_Group1/mini-project7/data'
try:
    X, y, class_labels = preprocess_dataset(dataset_path)
    print(f"Processed {len(X)} images")
    print(f"Class labels: {class_labels}")
    
    # Save the preprocessed data
    np.save('D:/KhouryGithub/CS5330_FA24_Group1/mini-project7/preprocessed_X.npy', X)
    np.save('D:/KhouryGithub/CS5330_FA24_Group1/mini-project7/preprocessed_y.npy', y)
    print("Preprocessed data saved successfully")
except Exception as e:
    print(f"Error during preprocessing: {str(e)}")