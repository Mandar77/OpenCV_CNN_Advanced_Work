import cv2
import numpy as np
import time
from threading import Thread

class VideoCaptureThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.ret:
                self.stop()
            else:
                self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

def orb_feature_detection(frame):
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints, descriptors = orb.detectAndCompute(frame, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def resize_frame(frame, scale_percent=50):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def stitch_images(img1, img2):
    try:
        kp1, des1 = orb_feature_detection(img1)
        kp2, des2 = orb_feature_detection(img2)

        matches = match_features(des1, des2)

        if len(matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None and np.isfinite(M).all():
                h, w = img2.shape[:2]
                result = cv2.warpPerspective(img1, M, (w * 3, h))
                result[0:h, 0:w] = img2
                return result

        print("Insufficient matches for stitching.")
        return None
    except cv2.error as e:
        print(f"OpenCV error during stitching: {str(e)}")
        return None

def create_panorama(frames):
    if len(frames) < 2:
        return frames[0] if frames else None
    
    panorama = frames[0]
    for frame in frames[1:]:
        stitched = stitch_images(panorama, frame)
        if stitched is not None:
            panorama = stitched
        else:
            print("Failed to stitch an image, continuing with current panorama...")
    return panorama

def is_frame_different(frame1, frame2, threshold=1000):
    frame1_resized = resize_frame(frame1)
    frame2_resized = resize_frame(frame2)
    
    if frame1_resized.shape != frame2_resized.shape:
        return True
    
    diff = cv2.absdiff(frame1_resized, frame2_resized)
    non_zero_count = np.count_nonzero(diff)
    return non_zero_count > threshold

def main():
    video_stream = VideoCaptureThread().start()
    time.sleep(1.0)  # Allow camera to warm up
    frames = []
    capturing = False
    start_time = time.time()
    frame_count = 0
    last_capture_time = 0
    capture_interval = 0.5

    while True:
        frame = video_stream.read()
        
        if frame is None:
            break
        
        frame_count += 1
        elapsed_time = time.time() - start_time

        # Check to prevent division by zero
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        else:
            fps = 0

        if capturing and time.time() - last_capture_time > capture_interval:
            resized_frame = resize_frame(frame.copy(), scale_percent=50) 
            if not frames or is_frame_different(frames[-1], resized_frame):
                frames.append(resized_frame)
                last_capture_time = time.time()

        # Display FPS with smaller font size
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

        window_name_frame = 'Frame'
        cv2.namedWindow(window_name_frame, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name_frame, 800, 600)
        cv2.imshow(window_name_frame, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            capturing = True
            frames.clear()
            last_capture_time = time.time()
            print("Started capturing frames for panorama")
        elif key == ord('a'):
            capturing is False
            print("Stopped capturing frames. Creating panorama...")
            if len(frames) > 1:
                panorama = create_panorama(frames)
                if panorama is not None:
                    window_name_panorama = 'Panorama'
                    cv2.namedWindow(window_name_panorama, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name_panorama, 1200, 600)
                    cv2.imshow(window_name_panorama, panorama)
                    cv2.imwrite('panorama.jpg', panorama)
                    print("Panorama saved as 'panorama.jpg'")
                else:
                    print("Failed to create panorama")
            else:
                print("Not enough frames captured for panorama")
        elif key == ord('q'):
            break

    video_stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()