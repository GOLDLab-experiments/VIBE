# extern the module classes
from .EmotionRecognizer import EmotionEstimator
from .EmotionRecognizer import FaceInferenceResults

__all__ = ["EmotionEstimator", "FaceInferenceResults"]

# check if the emotion recognition model is available
import os
REQUIRED_MODELS = [
    "intel/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml",
]
def _check_models():
    missing = [m for m in REQUIRED_MODELS if not os.path.exists(m)]
    if missing:
        raise RuntimeError(
            f"Required model files not found: {missing}\n"
            "Please run: python -m LLM.download_models to download the necessary models."
        )
_check_models()