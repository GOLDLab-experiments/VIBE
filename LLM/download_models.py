import subprocess
import sys
import os

required_models = [
    "emotions-recognition-retail-0003",
    "face-detection-retail-0004"
]

def ensure_openvino_dev():
    # Ensure openvino-dev is installed, install if missing
    try:
        import openvino
    except ImportError:
        print("openvino-dev not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openvino-dev"])

def download_models():
    # Download all models using omz_downloader with comma-separated model names
    model_names = ",".join(required_models)
    subprocess.check_call(["omz_downloader", "--name", model_names])

if __name__ == "__main__":
    # Run the model download process if this script is executed directly
    ensure_openvino_dev()
    download_models()