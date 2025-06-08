import cv2
import os
import subprocess
import sys

import numpy as np
from dotenv import load_dotenv
from openvino.runtime import Core
from dataclasses import dataclass

# --- FaceInferenceResults ---
@dataclass
class FaceInferenceResults:
    """
    Stores inference results for a detected face, including bounding box, confidence, and emotion attributes.
    """
    emotion: str | None = None
    emotion_scores: np.ndarray | None = None

# --- EmotionEstimator ---
class EmotionEstimator:
    """
    EmotionEstimator uses an OpenVINO model to recognize emotions from an image.
    """
    EMOTION_LABELS = [
        'neutral', 'happy', 'sad', 'surprise', 'anger'
    ]

    def __init__(self, core: Core, model_path: str, device_name: str = 'CPU'):
        model = core.read_model(model=model_path)
        self.compiled_model = core.compile_model(model=model, device_name=device_name)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

    def estimate(self, image: np.ndarray, results: FaceInferenceResults):
        _, _, H, W = self.input_layer.shape
        resized = cv2.resize(image, (W, H))
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        outputs = self.compiled_model([blob])[self.output_layer]
        scores = outputs[0]
        emotion_idx = int(np.argmax(scores))
        results.emotion = self.EMOTION_LABELS[emotion_idx]
        results.emotion_scores = scores

# --- EmotionPipeline ---
class EmotionPipeline:
    """
    EmotionPipeline runs emotion recognition on the entire image.
    """
    def __init__(self, core: Core, emotion_model_path: str, device_name: str = 'CPU'):
        self.emotion_estimator = EmotionEstimator(core, emotion_model_path, device_name)

    def process(self, image: np.ndarray) -> FaceInferenceResults:
        results = FaceInferenceResults()
        self.emotion_estimator.estimate(image, results)
        return results

# --- EmotionRecognizer ---
class EmotionRecognizer:
    def __init__(self, env_path='.env.emotion'):
        print("Initializing EmotionRecognizer...")
        load_dotenv(env_path)
        em_path = os.environ["EM_MODEL_PATH"]

        print(f"Using emotion model path: {em_path}")
        if not os.path.exists(em_path):
            print(f"Model file not found at {em_path}. Downloading models...")
            script_path = os.path.join(os.path.dirname(__file__), "download_models.py")
            subprocess.check_call([sys.executable, script_path])

        self.core = Core()
        self.pipeline = EmotionPipeline(self.core, em_path)
        cam_source = os.environ.get("CAMERA_SOURCE", "0")
        if cam_source.isdigit():
            self.cap = cv2.VideoCapture(int(cam_source))
        else:
            raise ValueError(f"Invalid camera source: {cam_source}")
        self.emotions_array = ['neutral', 'happy', 'sad', 'surprise', 'anger']

    def detect(self, frame):
        results = self.pipeline.process(frame)
        emotion_prob = results.emotion_scores
        detections = [
            (self.emotions_array[index], round(prob[0][0], 3))
            for index, prob in enumerate(emotion_prob)
        ]
        # Optionally draw label:
        # if results.emotion:
        #     label = f"{results.emotion}"
        #     cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        return frame, detections

    def run(self):
        while True:
            frame = None
            if self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    break
            if frame is None:
                print("No frame captured. breaking...")
                break
            frame, detections = self.detect(frame)
            print("Detections:", detections)
            cv2.imshow("Emotion", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if not self.cap:
                cv2.waitKey(0)
                break
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    recognizer = EmotionRecognizer()
    recognizer.run()

if __name__ == "__main__":
    main()