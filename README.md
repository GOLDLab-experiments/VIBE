# VIBE - Visual Intelligence-Based Engine

## 🧠 Overview

**VIBE** is a sophisticated scene analysis and classification engine that processes visual data from a camera feed to understand the context of a scene. By combining state-of-the-art machine learning models, VIBE can identify objects, recognize human emotions, and analyze body posture to classify scenes as **"Benign," "Malicious," or "Authorized."**

This project is part of a larger system responsible for controlling a robot’s movements.  
👉 For full system details and installation instructions, see: **[Link to main project]**

---

## ⚙️ How It Works

The system follows a structured pipeline:

1. **Frame Capture**  
   A single image frame is captured from the video stream.

2. **Parallel Processing**  
   The frame is sent simultaneously to three specialized modules:  
   - Object Detection  
   - Emotion Recognition  
   - Skeleton Detection  

3. **Data Aggregation**  
   Outputs from the modules—objects, emotions, and skeletal landmarks—are collected and combined.

4. **LLM-Based Analysis**  
   The combined data is formatted into a prompt and sent to a **Large Language Model (LLM)** for interpretation.

5. **Scene Classification**  
   The LLM analyzes the prompt and returns a final scene label along with a rationale.

---

## 🧩 Components

### 1. LLM Orchestrator (`LLM.py`)
The central controller of VIBE. It coordinates all other components and uses the `Qwen/Qwen3-0.6B` LLM for prompt-based scene classification. Few-shot prompting is used for consistency and accuracy.

### 2. Object Detection (`ObjectDetection/ObjectDetector.py`)
Uses a YOLOv8 model (`yolov8l-oiv7.pt`) to detect objects in the scene. It outputs object labels and confidence scores to the orchestrator.

### 3. Emotion Recognition (`EmotionRecognition/EmotionRecognizer.py`)
Utilizes the OpenVINO toolkit for fast emotion recognition. It detects faces and classifies expressions into categories such as *neutral*, *happy*, *sad*, *surprised*, and *angry*.

### 4. Skeleton Detection (`skeleton.py`)
Employs the MediaPipe library to detect human poses by extracting key skeletal landmarks—crucial for interpreting posture and body language.

### 5. Model Downloader (`download_models.py`)
A utility script to automatically download all required models for the Emotion Recognition module, simplifying setup.

---

## 👥 Authors

- Gali Tal  
- Arbel Tepper
