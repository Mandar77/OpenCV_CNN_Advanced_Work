import cv2
import numpy as np
import time
import os
import argparse

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(edges):
    mask = np.zeros_like(edges)
    height, width = edges.shape
    polygon = np.array([[0, height], [width, height], [width//2, height//2]], np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

def hough_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=200)
    return lines

def average_slope_intercept(lines):
    left_lines = []
    right_lines = []
    if lines is None:
        return None, None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))
    
    left_avg = np.average(left_lines, axis=0) if len(left_lines) > 0 else None
    right_avg = np.average(right_lines, axis=0) if len(right_lines) > 0 else None
    return left_avg, right_avg

def create_lines(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [(x1, y1, x2, y2)]

def draw_lane_lines(frame, lines):
    line_image = np.zeros_like(frame)
    if lines is not None:
        height, width, _ = frame.shape
        y1 = height
        y2 = int(height * 0.6)
        left_avg, right_avg = average_slope_intercept(lines)
        left_line = create_lines(y1, y2, left_avg)
        right_line = create_lines(y1, y2, right_avg)
        
        for line in [left_line, right_line]:
            if line:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Red lanes
    return cv2.addWeighted(frame, 0.8, line_image, 1, 0)

def process_video(input_file, output_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    vs = cv2.VideoCapture(input_file)
    if not vs.isOpened():
        raise IOError(f"Could not open video file: {input_file}")

    width = int(vs.get(3))
    height = int(vs.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height), True)

    if not out.isOpened():
        raise IOError(f"Could not create output file: {output_file}")

    frame_count = 0
    while True:
        ret, frame = vs.read()
        if not ret:
            break

        edges = preprocess(frame)
        roi = region_of_interest(edges)
        lines = hough_lines(roi)
        lane_frame = draw_lane_lines(frame, lines)

        out.write(lane_frame)
        frame_count += 1

        cv2.imshow("Lane Detection", lane_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()
    out.release()
    cv2.destroyAllWindows()

    return f"Processed {frame_count} frames"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video file path or camera input")
    parser.add_argument("-f", "--file", type=str, help="Path to the video file")
    parser.add_argument("-o", "--out", type=str, help="Output video file name")
    args = parser.parse_args()

    if args.file and args.out:
        try:
            result = process_video(args.file, args.out)
            print(f"Successfully processed {args.file}: {result}")
        except Exception as e:
            print(f"Error processing {args.file}: {str(e)}")
    else:
        print("Please provide both input and output file names using -f and -o arguments.")
        print("Example: python script.py -f input_video.mp4 -o output_video.avi")