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

### 1. LLM Orchestrator (`LLM/LLM.py`)
The central controller of VIBE. It coordinates all other components and uses the `Qwen/Qwen3-0.6B` LLM for prompt-based scene classification. Few-shot prompting is used for consistency and accuracy.

### 2. Object Detection (`ObjectDetection/ObjectDetector.py`)
Uses a YOLOv8 model (`yolov8l-oiv7.pt`) to detect objects in the scene. It outputs object labels and confidence scores to the orchestrator.

### 3. Emotion Recognition (`EmotionRecognition/EmotionRecognizer.py`)
Utilizes the OpenVINO toolkit for fast emotion recognition. It detects faces and classifies expressions into categories such as *neutral*, *happy*, *sad*, *surprised*, and *angry*.

### 4. Skeleton Detection (`skeleton/skeleton.py`)
Employs the MediaPipe library to detect human poses by extracting key skeletal landmarks—crucial for interpreting posture and body language.

### 5. Model Downloader (`LLM/download_models.py`)
A utility script to automatically download all required models for the LLM module, simplifying setup and ensuring all necessary dependencies are available.

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- A camera (webcam or external camera)
- CUDA-compatible GPU (optional, for faster inference)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd VIBE
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the project**
   ```bash
   pip install -e .
   ```

4. **Download required models**
   The emotion recognition models should be automatically available in the `intel/` directory. If not, ensure the following structure exists:
   ```
   intel/
   ├── face-detection-retail-0004/FP16/
   └── emotions-recognition-retail-0003/FP16/
   ```

5. **Configure environment variables**
   The system uses a `.env.emotion` file for configuration. Ensure it contains:
   ```bash
   FD_MODEL_PATH=intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml
   EM_MODEL_PATH=intel/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml
   CAMERA_SOURCE=0
   ```

---

## 🚀 Usage

### Running the Main System

To start the VIBE engine:

```bash
python -m LLM.LLM
```

### System Operation

1. **Camera Initialization**: The system will initialize your camera and display a live feed
2. **Capture**: Press the **SPACEBAR** to capture an image for analysis
3. **Processing**: The system will:
   - Detect objects in the scene
   - Recognize emotions from faces
   - Extract skeletal landmarks
   - Send combined data to the LLM for classification
4. **Results**: View the scene classification and reasoning in the terminal
5. **Exit**: Press **ESC** to quit the application

### Example Output

```
Objects detected: [('Person', 0.95), ('Chair', 0.82), ('Table', 0.76)]
Emotion probabilities: [('neutral', 0.65), ('happy', 0.25), ('sad', 0.10)]
Skeleton landmarks detected: 33 points
Scene Classification: Benign
Reasoning: The scene shows a person in a relaxed posture with neutral expression in a typical indoor environment.
```

### Individual Module Testing

You can also test individual components:

- **Object Detection**: `python -m ObjectDetection.ObjectDetector`
- **Emotion Recognition**: `python -m motionRecognition.EmotionRecognizer`
- **Skeleton Detection**: `python -m skeleton.skeleton`

---

## 🔧 Configuration

### Environment Variables

- `FD_MODEL_PATH`: Path to face detection model
- `EM_MODEL_PATH`: Path to emotion recognition model  
- `CAMERA_SOURCE`: Camera index (usually 0 for default camera)

### Model Customization

- Replace YOLOv8 models in `ObjectDetection/` for different object detection capabilities
- Modify the LLM model in `LLM.py` by changing the model name in the initialization

---

## 🐛 Troubleshooting

### Common Issues

1. **Camera not found**: Ensure your camera is connected and not used by another application
2. **Model files missing**: Check that all model files exist in the specified paths, use ```python -m LLM.download_models```
3. **Import errors**: Make sure the project is installed with `pip install -e .`
4. **CUDA errors**: Install PyTorch with CUDA support or set device to 'cpu'

### Performance Tips

- Use CUDA-enabled PyTorch for faster LLM inference
- Adjust camera resolution in the code if needed
- Consider using lighter model variants for real-time performance

---

## 👥 Authors

- [Gali Tal](https://github.com/galital5)
- [Arbel Tepper](https://github.com/ArbelTepper)
