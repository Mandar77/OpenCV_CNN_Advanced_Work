import cv2
import numpy as np
import time

# Open Video Camera
vs = cv2.VideoCapture(0)
time.sleep(2.0)
frame_count = 0
start_time = time.time()
fps = 0

# Modes dictionary to store the state of each mode
modes = {
    "translation": False,
    "rotation": False,
    "scaling": False,
    "perspective_transformation": False,

}

# Helper function to reset all modes
def reset_modes():
    for key in modes:
        modes[key] = False

# Mode handling based on key presses
def handle_key_press(key):
    mode_keys = {
        "t": "translation",
        "r": "rotation",
        "s": "scaling",
        "p": "perspective_transformation",
    }
    
    if key in mode_keys:
        mode = mode_keys[key]
        if not modes[mode]:
            reset_modes()  # Reset all other modes
            modes[mode] = True
        else:
            modes[mode] = False

# Calculate and display fps
def calculate_and_display_fps(frame, frame_count, start_time, fps):
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Update FPS every second
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        start_time = current_time
        frame_count = 0

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame, frame_count + 1, start_time, fps
    
# Main loop to process video frames
while True:
    ret, frame = vs.read()
    if not ret:
        break
    
    # Create a copy of the frame for modifications
    modified_frame = frame.copy()

    # Apply the appropriate mode based on active flags to the modified frame
    if modes["translation"]:
        tx, ty = 50, 50  # Translation values
        M = np.float32([[1, 0, tx], [0, 1, ty]])  # Translation matrix

    # Apply translation
        modified_frame = cv2.warpAffine(modified_frame, M, 
                              (modified_frame.shape[1], modified_frame.shape[0]))
    if modes["rotation"]:
        (h, w) = modified_frame.shape[:2]  # First two values represent height and width
        center = (0, 0)  # Rotation around the top-left corner
        angle = 45  # Rotate by 45 degrees

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Perform the rotation and update modified_frame
        modified_frame = cv2.warpAffine(modified_frame, M, (w, h))
    
    
    if modes["scaling"]:

        M = np.float32([[1.5, 0, 0], [0, 1.5, 0]])  # Translation matrix

    # Apply translation
        modified_frame = cv2.warpAffine(modified_frame, M, 
                              (modified_frame.shape[1], modified_frame.shape[0]))
    if modes["perspective_transformation"]:
        pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250], [250, 200]])
        #print("hello")

        # Get projective transformation matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)

        # Apply projective transformation
        modified_frame = cv2.warpPerspective(modified_frame, M, 
                                              (modified_frame.shape[1], modified_frame.shape[0]))

    frame, frame_count, start_time, fps = calculate_and_display_fps(frame, frame_count, start_time, fps)

    # Concatenate the original and modified frames horizontally
    combined = cv2.hconcat([frame, modified_frame])

    # Display the concatenated frames
    cv2.imshow("Original (Left) | Modified (Right)", combined)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # Handle key press for modes
    handle_key_press(chr(key).lower())  # Convert key to lowercase for easier handling

    # Quit the loop if 'q' is pressed
    if key == ord("q"):
        break

# Release resources
vs.release()
cv2.destroyAllWindows()