import cv2
import numpy as np
import time

# print(f"OpenCV version: {cv2.__version__}")

class ObjectDetector:
    def __init__(self, config_file, weights_file, classes_file, target_classes, conf_threshold=0.5, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = (416, 416)
        self.classes = self.load_classes(classes_file)
        self.target_classes = target_classes
        self.net = self.load_network(config_file, weights_file)

    def load_classes(self, classes_file):
        with open(classes_file, 'rt') as f:
            return f.read().rstrip('\n').split('\n')

    def load_network(self, config_file, weights_file):
        net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        return [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255, self.input_size, [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.get_output_layers())
        return self.process_detections(frame, outputs)

    def process_detections(self, frame, outputs):
        frame_height, frame_width = frame.shape[:2]
        class_ids, confidences, boxes = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x, center_y = int(detection[0] * frame_width), int(detection[1] * frame_height)
                    width, height = int(detection[2] * frame_width), int(detection[3] * frame_height)
                    left, top = int(center_x - width/2), int(center_y - height/2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        detected_objects = []
        for i in indices:
            box = boxes[i]
            left, top, width, height = box
            self.draw_prediction(frame, class_ids[i], confidences[i], left, top, left + width, top + height)
            detected_objects.append(self.classes[class_ids[i]])

        return detected_objects

    def draw_prediction(self, frame, class_id, confidence, left, top, right, bottom):
        label = f'{self.classes[class_id]}: {confidence:.2f}'
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

class VideoRecorder:
    def __init__(self, save_fps=5):
        self.frames = []
        self.recording = False
        self.start_time = None
        self.save_fps = save_fps
        self.video_counter = 1

    def start_recording(self):
        self.recording = True
        self.start_time = time.time()
        self.frames = []
        print(f"Started recording video #{self.video_counter}...")

    def add_frame(self, frame):
        if self.recording:
            self.frames.append(frame)

    def stop_recording(self):
        if self.recording:
            self.recording = False
            output_filename = f"DetectedObject_{self.video_counter}.mp4"
            out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), self.save_fps, 
                                  (self.frames[0].shape[1], self.frames[0].shape[0]))
            for frame in self.frames:
                out.write(frame)
            out.release()
            print(f"Stopped recording and saved video as {output_filename}.")
            self.video_counter += 1

    def should_stop(self):
        return self.recording and (time.time() - self.start_time >= 5)

def main():
    detector = ObjectDetector("yolov3.cfg", "yolov3.weights", "coco.names", ["cup", "banana"])
    recorder = VideoRecorder()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        detected_objects = detector.detect_objects(frame)

        if any(obj in detector.target_classes for obj in detected_objects) and not recorder.recording:
            recorder.start_recording()

        recorder.add_frame(frame)

        if recorder.should_stop():
            recorder.stop_recording()

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()