# CS5330 Group 1 - Multi-Class Object Detection (Projects 9-1 and 9-2)

## Project 9-1: Multi-Class Object Detection with RCNN

### Step 1: Dataset Creation and Image Annotation
We added a "remotes" class to our dataset, using images from mini-project 7 (stored in the `remotes` folder). After installing **LabelMe**, we annotated these images with bounding boxes and saved the annotations as JSON files in the `images` folder.

### Step 2: Convert Annotations from JSON to CSV
To prepare the annotations for training, we used the `json_to_csv.py` script to convert JSON files to CSV format.

### Step 3: Modify RCNN Code for Multi-Class Training
We updated the provided RCNN code to handle both "remotes" and "airplanes" classes, saving it as `rcnn-multi.py`. Running this script loads the images and annotations to train the model. Given the dataset size and number of epochs, this process is time-consuming. Upon completion, a loss graph is saved in the folder along with sample test results.

### Step 4: Model Testing
We evaluated the model using `test_rcnn_6.py`, testing it on three random images from each class ("remotes" and "airplanes").

### Step 5: Upload Model and Data to Google Drive
Due to large file sizes, we uploaded the data and model to Google Drive. [[Model and Data](https://northeastern-my.sharepoint.com/:f:/r/personal/ambulkar_m_northeastern_edu/Documents/mini-project9?csf=1&web=1&e=5Zhica).]

### How to Run 9-1
1. Download the necessary data and model files from Google Drive.
2. Train the model:  
   ```
   python rcnn-multi.py
    ```
3. Test the model with sample data:
    ```
    python test_rcnn_6.py
    ```

## Project 9-2: Performance Optimization of the Multi-Class Object Detection Model

### Step 1: Initial Model Testing with Webcam Input
To evaluate the performance of our 9-1 model in a real-time setting, we used `WebCamSave-rcnn.py` to test the model with webcam input. The model accurately detected objects but had low FPS, resulting in laggy and delayed output.

### Step 2: Improving Model Efficiency with RCNN-Light
In 9-2, our focus was on optimizing the model to achieve faster inference times and higher FPS. We developed `rcnn-light.py`, a streamlined version of the model, aimed at enhancing speed without compromising detection accuracy. After modifying the code, we retrained this lightweight model and tested it to confirm performance gains.

### Step 3: Uploading Optimized Model and Data to Google Drive
Due to the large file sizes, we uploaded both the optimized model and associated data to Google Drive for easy access. [Models and Data](https://northeastern-my.sharepoint.com/:f:/r/personal/ambulkar_m_northeastern_edu/Documents/mini-project9?csf=1&web=1&e=5Zhica)

### How to Run 9-2
1. Test the initial model’s performance on webcam input:
   ```
   python WebCamSave-rcnn.py
    ```
2. Train the optimized model:
    ```
    python rcnn_light.py
    ```
3. Test the optimized model:
    ```
    python test_rcnn_light.py
    ```
4. Re run the webcam to compare results:
    ```
    python WebCamSave-rcnn.py 
    ```