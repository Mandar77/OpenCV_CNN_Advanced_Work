# Mini Project 10: U-Net Model for Vehicle Segmentation

## Project Overview
This project implements a U-Net architecture for vehicle segmentation using a custom dataset. The implementation includes data collection, preprocessing, mask generation, and model training with various optimizations.


## Project Structure

```
bash
mini-project10/
├── data/
│ └── raw/
│  ├── images/ # Contains vehicle images
│  ├── masks/ # Contains corresponding segmentation masks
│  └── labels/ # Contains corresponding labels according yolov5 format
├── src/
│ ├── data/
│ │ └── make_dataset.py # Data preprocessing and loading
│ ├── models/
│ │ ├── models.py # custom U-Net model and training parameter defination
│ │ └── train_model.py # U-Net model and training logic
│ └── visualization/
│ └── visualize.py # Visualization utilities
├── config/
│ └── config.yaml # Configuration parameters
├── results/
│ ├── figures/ # Visualization outputs
│ └── models/ # Trained model checkpoints
├── logs/
│  ├── train/ 
│  └── validation/
└── main.py
└── generate_data.py
└── generate_labels.py
└── README.md
└── requirements.txt
```
## Dataset Creation
We created a custom dataset for vehicle segmentation:

1. **Data Collection**:
   - Downloaded over 500 vehicle images using Bing Image Downloader
   - Used multiple search queries for diverse vehicle images
   - Implemented automatic download and organization

2. **Data Processing**:
   - Organized images in `data/raw/images`
   - Generated corresponding masks using YOLO detection
   - Filtered images to focus on vehicle-only scenes
   - Current dataset size: ~543 image-mask pairs and corresponding labels

3. **YOLO Detection Results**:
   - Successfully detected vehicles (cars, trucks, buses)
   - Filtered out non-vehicle objects
   - Handled multiple vehicle instances per image
   - Generated binary masks for segmentation
   - Generated corresponding labels according to yolo format.

## Implementation Details

### 1. Data Pipeline
- **Image Processing**:

- Image Resized to (256,256)

- Mask Generation:

```
python
def create_vehicle_mask(image, detection):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for obj in detection:
        if obj['class'] in ['car', 'truck', 'bus']:
            # Create mask for vehicle
            cv2.drawContours(mask, [obj['contour']], -1, 255, -1)
    return mask
```

### 2. Model Architecture

Modified U-Net architecture with:
- Input size: 256x256x3
- Encoder: 4 conv blocks with max pooling
- Decoder: 4 upsampling blocks with skip connections
- Output: Binary segmentation mask

# Model Parameters
INPUT_IMAGE_WIDTH: 256
INPUT_IMAGE_HEIGHT: 256
NUM_CHANNELS: 3
NUM_CLASSES: 1

# Training Parameters
INIT_LR: 0.0003
NUM_EPOCHS: 150
BATCH_SIZE: 8

4. Optimizations Implemented
Data Augmentation:
Random rotation (±45°)
Horizontal flips
Brightness variation
Zoom range: 0.2
Model Improvements:
Batch normalization
Dropout layers (rate=0.3)
Skip connections
Dice loss function
Training Enhancements:
Learning rate scheduling
Early stopping
Model checkpointing
Setup Instructions
Environment Setup:
bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

Dataset Generation:
bash
# Generate dataset
python src/data/generate_data.py

Model Training:
bash
# Train the model
python main.py

Current Results
Dataset Statistics:
Total images: 491
Training set: 393 (80%)
Validation set: 98 (20%)
Model Performance:
Training accuracy: ~60%
Validation accuracy: ~64%
Current limitations: Working on improving accuracy
Next Steps
Model Optimization:
Implement additional data augmentation
Try different model architectures
Experiment with loss functions
Data Quality:
Improve mask generation
Filter dataset for better quality
Add more diverse vehicle images
Training Improvements:
GPU acceleration
Cross-validation
Ensemble methods
Dependencies
Python 3.8+
TensorFlow 2.x
OpenCV
NumPy
Matplotlib
Ultralytics YOLO
Bing Image Downloader
References
U-Net Paper: U-Net: Convolutional Networks for Biomedical Image Segmentation
YOLO Documentation: Ultralytics YOLOv8