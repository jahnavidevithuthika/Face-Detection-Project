# Face Detection Project

This project detects human faces in real time using Python and OpenCV.

## Features
- Real-time face detection
- Webcam support
- AI-based face recognition box
- Simple beginner-friendly project

## Technologies Used
- Python
- OpenCV

## Project Structure

Face-Detection-Project
│
├── face_detect.py
├── retina_detect.py
├── requirements.txt
└── README.md

## Installation

Install required library:

```bash
pip install opencv-python
```

## Run Project

```bash
python face_detect.py
```

## Retina Detection

Use the retina/iris detection script:

```bash
python retina_detect.py
```

This version detects a human face and then looks for eye regions and estimates the iris (retina-like region) using OpenCV circle detection.

## How It Works
- Opens webcam
- Detects face using OpenCV Haar Cascade
- Finds eyes within the face region
- Estimates the iris using Hough Circles
- Draws a green circle around the detected retinal area

## Author
Jahnavi