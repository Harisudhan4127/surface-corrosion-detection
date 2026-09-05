# Driver Drowsiness Detection GUI

A standalone GUI application for driver drowsiness prediction using OpenCV and Tkinter.

## Features
- Load and predict on a single image
- Start webcam prediction
- Start a video stream from a camera ID or URL
- Visual feedback with bounding boxes and status label

## Install
```bash
pip install -r driver_drowsiness_requirements.txt
```

## Run
```bash
python driver_drowsiness_gui.py
```

## Usage
1. Click `Open Image`, choose a photo, and then click `Predict Image`.
2. For live evaluation, click `Start Webcam` or enter a stream URL and `Start Stream`.
3. Click `Stop` to end the camera session.

## Notes
- This version uses OpenCV Haar cascades for face and eye detection.
- Drowsiness is estimated when eye detection fails on consecutive frames.
- The script is designed for local prediction without requiring the archived YOLO model or PyQt5.
