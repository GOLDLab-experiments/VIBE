# VIBE - Visual Intelligence-Based Engine

## Overview

VIBE is a sophisticated scene analysis and classification engine. It processes visual data from a camera feed to understand the context of a scene. By leveraging a combination of state-of-the-art machine learning models, VIBE can identify objects, recognize human emotions, and analyze body posture to classify a scene as "Benign," "Malicious," or "Authorized."

This project is a component of a larger system which controlls the robot's movements. For more details and installation instructions, please visit: [Link to main project]

## How It Works

The system operates in a pipeline:

1.  **Frame Capture**: An image frame is captured from a video stream.
2.  **Parallel Processing**: The frame is simultaneously fed into three specialized modules:
    *   Object Detection
    *   Emotion Recognition
    *   Skeleton Detection
3.  **Data Aggregation**: The outputs from these modules—a list of detected objects, recognized emotions, and key skeletal landmarks—are collected.
4.  **LLM-Based Analysis**: The aggregated data is formatted into a detailed prompt and sent to a Large Language Model (LLM).
5.  **Scene Classification**: The LLM analyzes the prompt and provides a final classification for the scene, along with a justification for its decision.

## Components

### 1. LLM Orchestrator (`LLM.py`)

This is the central nervous system of VIBE. It orchestrates the other components and uses a powerful Large Language Model (`Qwen/Qwen3-0.6B`) to interpret the combined data. It uses few-shot prompting to guide the LLM into providing a consistent and accurate classification.

### 2. Object Detection (`ObjectDetection/ObjectDetector.py`)

This module uses a YOLO (You Only Look Once) model (`yolov8l-oiv7.pt`) to detect and identify various objects within the captured frame. The list of detected objects and their confidence scores are passed to the orchestrator.

### 3. Emotion Recognition (`EmotionRecognition/EmotionRecognizer.py`)

This component utilizes the OpenVINO toolkit to perform high-speed emotion recognition. It identifies faces in the image and classifies their expressions into categories such as 'neutral', 'happy', 'sad', 'surprise', or 'anger'.

### 4. Skeleton Detection (`skeleton.py`)

Using the MediaPipe library, this module detects human poses by identifying key skeletal landmarks. This provides insights into body language and posture, which is a crucial element for contextual understanding.

### 5. Model Downloader (`download_models.py`)

This utility script ensures that all the necessary models for the Emotion Recognition module are downloaded and available, making the setup process smoother.

## Authors

*   Gali Tal
*   Arbel Tepper
