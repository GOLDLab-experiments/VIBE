import os
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Minimal: simple registry of model options you can change in code.
# Choose one key below (local YOLO weights or HF). Add/remove entries as needed.
AVAILABLE_MODELS = {
    # Local weights already in repo
    "oiv7_large": {"local": "yolov8l-oiv7.pt"},
    "oiv7_medium": {"local": "yolov8m-oiv7.pt"},
    "oiv7_small": {"local": "yolov8s-oiv7.pt"},
    # Hugging Face (optional). Requires: pip install huggingface_hub
    "hf_nano": {"hf_repo": "ultralytics/ultralytics", "hf_filename": "yolov8n.pt"},
    "hf_small": {"hf_repo": "ultralytics/ultralytics", "hf_filename": "yolov8s.pt"},
    # Added safety helmet detection model (assumed weights file 'best.pt'). Change filename if different.
    "hf_helmet": {"hf_repo": "sharathhhhh/safetyHelmet-detection-yolov8", "hf_filename": "best.pt"},
}

class ObjectDetector:
    def __init__(self, model_key: str = "oiv7_large", model_name: str | None = None,
                 hf_repo: str | None = None, hf_filename: str | None = None):
        """
        Initialize YOLO detector.
        Easiest way to switch: change model_key (or set env OBJECT_DETECTOR_MODEL).
        Args:
            model_key: key from AVAILABLE_MODELS (overridden by env OBJECT_DETECTOR_MODEL if present).
            model_name / hf_repo / hf_filename: explicit overrides (advanced; normally leave None).
        """
        print("Initializing ObjectDetector...")
        # Allow environment override without touching code
        env_key = os.getenv("OBJECT_DETECTOR_MODEL")
        if env_key:
            model_key = env_key
        if model_key not in AVAILABLE_MODELS and not (model_name or hf_repo):
            raise ValueError(f"Unknown model_key '{model_key}'. Available: {list(AVAILABLE_MODELS.keys())}")

        cfg = AVAILABLE_MODELS.get(model_key, {})
        # Priority: explicit args > registry entry
        model_name = model_name or cfg.get("local")
        hf_repo = hf_repo or cfg.get("hf_repo")
        hf_filename = hf_filename or cfg.get("hf_filename")

        self.model_path = self._resolve_weights(model_name, hf_repo, hf_filename)
        print(f"Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.labels = self.model.names

    def _resolve_weights(self, model_name: str | None, hf_repo: str | None, hf_filename: str | None) -> str:
        if hf_repo and hf_filename:
            if hf_hub_download is None:
                raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub or pick a local model key.")
            print(f"Downloading from Hugging Face {hf_repo}/{hf_filename} ...")
            return hf_hub_download(repo_id=hf_repo, filename=hf_filename)
        if not model_name:
            raise ValueError("No local model_name provided and Hugging Face parameters missing.")
        return model_name

    def detect(self, image):  # numpy array / cv2 frame accepted
        res = self.model.predict(image, conf=0.2, save=True, show=False, verbose=False)[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.data is None or len(boxes) == 0:
            return []
        detections = []
        for b in boxes:
            cls_id = int(b.cls.item()) if hasattr(b, 'cls') else int(b.data[0,5].item())
            conf = float(b.conf.item()) if hasattr(b, 'conf') else float(b.data[0,4].item())
            label = self.labels.get(cls_id, str(cls_id)) if isinstance(self.labels, dict) else self.labels[cls_id]
            detections.append((label, round(conf, 3)))
        return detections

def main():
    # To change model just modify the key here (or export OBJECT_DETECTOR_MODEL=hf_nano before running)
    object_detector = ObjectDetector(model_key="hf_helmet")
    # Example alternative (uncomment): object_detector = ObjectDetector(model_key="hf_nano")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    print("Press SPACE to capture an image, or ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)
        if key % 256 == 27:  # ESC
            print("Escape hit, closing...")
            break
        elif key % 256 == 32:  # SPACE
            image = frame.copy()
            detections = object_detector.detect(image)
            print("Detections:", detections)
            for label, prob in detections:
                cv2.putText(image, f"{label}: {prob}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Detections", image)
            cv2.waitKey(0)
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

"""
Example output:

Loading model yolov8s-oiv7.pt...
Running inference...
Detections: [('Human face', 0.764), ('Woman', 0.524)]

Loading model yolov8l-oiv7.pt...
Running inference...
Detections: [('Remote control', 0.615), ('Headphones', 0.496), ('Pen', 0.387), ('Fashion accessory', 0.367), ('Sock', 0.276)]
"""
