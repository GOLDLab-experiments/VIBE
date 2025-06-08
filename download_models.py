import subprocess
import sys
import os

def ensure_openvino_dev():
    # Ensure openvino-dev is installed, install if missing
    try:
        import openvino
    except ImportError:
        print("openvino-dev not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openvino-dev"])

def download_models():
    # Download all models listed in models.lst using omz_downloader
    models_lst = os.path.join(os.path.dirname(__file__), "models.lst")
    if not os.path.exists(models_lst):
        print(f"models.lst not found at {models_lst}")
        sys.exit(1)
    subprocess.check_call(["omz_downloader", "--list", models_lst])

if __name__ == "__main__":
    # Run the model download process if this script is executed directly
    ensure_openvino_dev()
    download_models()