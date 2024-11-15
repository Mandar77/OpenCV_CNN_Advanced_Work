import cv2
import numpy as np
import time
import argparse
import tensorflow as tf

# Configuration
MODEL_PATH = 'multi_class_detector.h5'
CONFIDENCE_THRESHOLD = 0.85
NMS_THRESHOLD = 0.5
MAX_PROPOSALS = 15
FRAME_SKIP = 2
RESIZE_FACTOR = 0.5

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-time object detection from video or camera")
    parser.add_argument("-i", "--input", type=str, help="Path to the input video file")
    parser.add_argument("-o", "--output", type=str, help="Path to the output video file")
    return parser.parse_args()

def apply_non_max_suppression(boxes, threshold):
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

def generate_proposals(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return ss.process()[:MAX_PROPOSALS]

def process_frame(frame, model):
    small_frame = cv2.resize(frame, (int(frame.shape[1] * RESIZE_FACTOR), int(frame.shape[0] * RESIZE_FACTOR)))
    proposals = generate_proposals(small_frame)
    
    boxes = []
    labels = []
    
    for x, y, w, h in proposals:
        x, y, w, h = map(lambda v: int(v / RESIZE_FACTOR), (x, y, w, h))
        region = frame[y:y+h, x:x+w]
        resized = cv2.resize(region, (224, 224))
        prediction = model.predict(np.expand_dims(resized, axis=0))[0]
        
        label = "Controller" if np.argmax(prediction) == 1 else "Aircraft"
        score = prediction[1] if label == "Controller" else prediction[0]
        
        if score > CONFIDENCE_THRESHOLD:
            boxes.append([x, y, x+w, y+h, score])
            labels.append(label)
    
    return np.array(boxes), labels

def draw_detections(frame, boxes, labels):
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2, score = map(int, box[:5])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    args = parse_arguments()
    model = tf.keras.models.load_model(MODEL_PATH)
    
    cap = cv2.VideoCapture(args.input if args.input else 0)
    width, height = int(cap.get(3)), int(cap.get(4))
    
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, 20.0, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue
        
        boxes, labels = process_frame(frame, model)
        nms_indices = apply_non_max_suppression(boxes, NMS_THRESHOLD)
        
        nms_boxes = boxes[nms_indices]
        nms_labels = [labels[i] for i in nms_indices]
        
        frame = draw_detections(frame, nms_boxes, nms_labels)
        
        fps = frame_count / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if out:
            out.write(frame)
        
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()