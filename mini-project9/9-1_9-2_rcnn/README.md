# CS5330 Group 1 - Project 9-1, 9-2 README: Multi-class Detection using RCNN

## 9-1 : Developing a Multi-Class Object Recognition Model
1. Dataset Expansion and Annotation:
- We enhanced our dataset by incorporating a "remotes" category, utilizing images from our previous work (located in the remotes directory). We employed LabelMe to annotate these images, creating bounding boxes and storing the annotations as JSON files in the dataset folder.
2. Annotation Format Conversion
- To prepare our data for model training, we developed the json_to_csv.py script, which transforms JSON annotations into the YOLO format.
3. Multi-Class Recognition Model Development
- We adapted the provided single-class detection code to accommodate both "remotes" and "airplane" classes, resulting in rcnn-multi.py. This script handles data loading, model training, and evaluation. Due to the extensive dataset and training iterations, this process is computationally intensive. Upon completion, the script generates a performance graph and example detection results.
5. Model and Data Storage
Given the large file sizes, we've stored our model and associated data in a cloud repository. Access Model and Data Here

### Running Phase A

1. Retrieve the necessary files from the cloud repository.
2. Train the multi-class model:

`python multi_class_detector.py`

3. Evaluate the model:

`python evaluate_model.py`

## Phase B: Optimizing Detection Performance
1. Real-time Performance Assessment
- To gauge our 9-1 model's real-time capabilities, we utilized WebCamSave-rcnn.py for webcam-based testing. While object detection was accurate, the frame rate was suboptimal, resulting in noticeable lag.
2. Model Optimization with LightDetector
- In 9-2, we focused on enhancing the model's efficiency to achieve faster inference and higher frame rates. We developed rcnn_light.py, a streamlined version of our original model, designed to boost speed without sacrificing detection accuracy. After implementation, we retrained this optimized model and conducted tests to verify performance improvements.
3. Cloud Storage of Optimized Model
- Due to file size constraints, we've uploaded both the optimized model and its associated data to our cloud repository. Access Optimized Model and Data

### Running Phase B
1. Test the initial model's real-time performance:

`python WebCamSave-rcnn.py`

2. Train the optimized model:

`python light_detector.py`

3. Evaluate the optimized model:

`python test_rcnn.py`

4. Reassess real-time performance:

`python WebCamSave-rcnn.py`