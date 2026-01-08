# Gesture-Controlled Drawing Application

---
## Authors
- Wiktor Rapacz
- Hanna Paczoska

---

## Problem Description
The goal of this project is to build an interactive application that detects hand gestures in real time using a webcam and maps them to drawing actions.  
The challenge is to design a gesture-based interface that is intuitive, stable, and works without any physical input devices such as a mouse or keyboard.

---
## Solution Overview
The application uses computer vision techniques to detect hand landmarks from a live webcam stream and interprets specific hand gestures as drawing commands.  
Based on the recognized gesture, the program draws geometric shapes on a virtual canvas, changes their size and color, or clears the screen.

The solution relies on a pre-trained hand landmark detection model and does not require training a custom machine learning model.

---
## Principle of Operation
1. The webcam captures a live video stream.
2. MediaPipe Hands detects hand landmarks (key points of the fingers and palm).
3. A simple rule-based gesture classifier determines which fingers are extended.
4. Recognized gestures are mapped to drawing actions:
   - OPEN_PALM → draw a circle
   - V_SIGN → draw a square
   - FIST → clear the canvas
   - INDEX_ONLY → increase shape size and change color (while holding)
   - PINKY_ONLY → decrease shape size (while holding)
5. The camera image and drawing canvas are blended and displayed in real time.
---
## Environment Setup Instructions
1. Install Python **3.10.x**
2. Create and activate a virtual environment
3. Install required libraries:

   pip install opencv-python mediapipe numpy
4. Ensure that a working webcam is connected to the system.
---
## Usage Instructions

1. Run the application:

    python gesture_draw_shapes.py

2. Place your right hand in front of the camera.

3. Use the following gestures:

        Open palm → draw a circle
        Two fingers (V sign) → draw a square
        Closed fist → clear the canvas
        Only index finger → increase shape size and change color
        Only pinky finger → decrease shape size

4. Press Q or ESC to exit the application.
---
## Technologies and Tools Used

    Python 3.10
    OpenCV
    MediaPipe (pre-trained Hands model)
    NumPy
    Webcam (real-time video input)