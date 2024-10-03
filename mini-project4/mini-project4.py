import cv2
import time

cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)

sift = cv2.SIFT_create()

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

prev_time = 0

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if ret1 and ret2:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
 
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        # matches = flann.knnMatch(descriptors1, descriptors2, k=2)
 
        # # Apply ratio test (for filtering good matches)
        # good_matches = []
        # for m, n in matches:
        #     if m.distance < 0.7 * n.distance:
        #         good_matches.append(m)

        matched_frame = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, matches[:50], None, flags=2)

        matching_score = sum([m.distance for m in matches[:50]])
        cv2.putText(matched_frame, f'Score: {matching_score:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(matched_frame, f'FPS: {int(fps)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Feature Matching with Dual Webcam', matched_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()