# README for Image World App

## By Group 1

### Project Set-up

Prerequisites : python3.x , pip

Packages to install:
pip3 install opencv-python
pip3 install numpy
pip3 install matplotlib

### Controls

You can press specific keys on the keyboard to toggle different video effects. All effects are saved and captured in a dictionary called "mode".
Translation Mode: Press "t" to translate the video frame from 0,0 to 50,50.\

Rotation Mode: Press "r" to rotate the video frame 45degree using 0,0 as center\

Scaling Mode: Press "s" to scale the image to 1.5 times.\

Perspective Transformation Mode: Press "p" to apply perspective transformation from [[50, 50], [200, 50], [50, 200], [200, 200]] to [[10, 100], [200, 50], [100, 250], [250, 200]]\

Quit: Press "q" to close the video and exit the program.

### Functions

reset_modes : Resets all modes (translate, rotation, scaling, etc.) to False, ensuring that no mode is active. This is called whenever a new mode is selected to disable other modes.

handle_key_press : Maps keyboard inputs to the corresponding modes (translation, rotation, scaling, etc.). It checks if a mode is active or inactive and toggles it accordingly. If a mode is activated, all other modes are reset.

Main Loop : The main loop continuously captures video frames from the webcam, applies the currently active mode (if any), and displays the result in real-time. It also listens for key presses to control which mode to activate or deactivate.

### Video Demonstration of Application

Link to video: https://youtu.be/HYPBj_Ge30U