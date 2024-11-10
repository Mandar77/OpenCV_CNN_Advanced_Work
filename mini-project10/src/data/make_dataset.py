import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_and_prepare_data(config):
    """Load and prepare the dataset for training"""
    print("\nLoading and preparing data...")
    
    # Get file paths
    image_files = sorted([f for f in os.listdir(config['IMAGE_DATASET_PATH']) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    mask_files = sorted([f for f in os.listdir(config['MASK_DATASET_PATH']) 
                        if f.endswith('_mask.png')])
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    images = []
    masks = []
    
    # Load and preprocess images and masks
    for img_file in image_files:
        mask_file = os.path.splitext(img_file)[0] + '_mask.png'
        if mask_file in mask_files:
            # Load image and mask
            img_path = os.path.join(config['IMAGE_DATASET_PATH'], img_file)
            mask_path = os.path.join(config['MASK_DATASET_PATH'], mask_file)
            
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None and mask is not None:
                # Resize
                img = cv2.resize(img, (config['INPUT_IMAGE_WIDTH'], 
                                     config['INPUT_IMAGE_HEIGHT']))
                mask = cv2.resize(mask, (config['INPUT_IMAGE_WIDTH'], 
                                       config['INPUT_IMAGE_HEIGHT']))
                
                # Normalize image
                img = img.astype(np.float32) / 255.0
                
                # Ensure mask is binary
                mask = (mask > 128).astype(np.float32)
                mask = np.expand_dims(mask, axis=-1)
                
                images.append(img)
                masks.append(mask)
    
    if not images:
        return None, None, None, None
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(masks)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=float(config['TEST_SPLIT']),
        random_state=42
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Testing set: {len(X_test)} images")
    
    return X_train, X_test, y_train, y_test

def create_data_generator(X_train, y_train, config):
    """Create data generator with augmentation"""
    data_gen_args = dict(
        rotation_range=float(config['ROTATION_RANGE']),
        width_shift_range=float(config['WIDTH_SHIFT_RANGE']),
        height_shift_range=float(config['HEIGHT_SHIFT_RANGE']),
        zoom_range=float(config['ZOOM_RANGE']),
        horizontal_flip=bool(config['HORIZONTAL_FLIP']),
        fill_mode='reflect',
        brightness_range=[float(x) for x in config['BRIGHTNESS_RANGE']]
    )
    
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed for both generators
    seed = 42
    image_generator = image_datagen.flow(
        X_train,
        batch_size=int(config['BATCH_SIZE']),
        seed=seed
    )
    mask_generator = mask_datagen.flow(
        y_train,
        batch_size=int(config['BATCH_SIZE']),
        seed=seed
    )
    
    return zip(image_generator, mask_generator)

def verify_data(X_train, X_test, y_train, y_test):
    """Verify data shapes and values"""
    print("\nData Verification:")
    print(f"Training images shape: {X_train.shape}")
    print(f"Training masks shape: {y_train.shape}")
    print(f"Testing images shape: {X_test.shape}")
    print(f"Testing masks shape: {y_test.shape}")
    
    # Verify value ranges
    print(f"\nValue ranges:")
    print(f"Images: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"Masks: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    return True