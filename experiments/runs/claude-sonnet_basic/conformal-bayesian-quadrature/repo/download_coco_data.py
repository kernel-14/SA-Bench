"""
Download and prepare MS-COCO data for the experiment.

This script downloads the pre-computed model predictions from the
learn-then-test repository (Bates et al., 2021), which is the same
data used in Angelopoulos & Bates (2023, Section 5.1).

The data consists of:
- Softmax scores from a ResNet-101 model trained on MS-COCO
- Binary labels for each of the 80 COCO categories

Total dataset size: 4952 examples (1000 calibration + 3952 test per trial)
"""

import os
import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


def download_file(url, dest_path):
    """Download a file from URL to dest_path."""
    print(f"Downloading {url} -> {dest_path}")
    
    if HAS_REQUESTS:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    elif HAS_URLLIB:
        urllib.request.urlretrieve(url, dest_path)
    else:
        raise RuntimeError("Neither requests nor urllib available")
    
    print(f"Downloaded {dest_path}")


def download_coco_data(data_dir="data/coco"):
    """
    Download MS-COCO model predictions.
    
    The data is from the learn-then-test repository:
    https://github.com/aangelopoulos/ltt
    
    Specifically, we use the COCO multilabel classification data
    from the conformal risk control experiments.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # The data is available from the conformal risk control repository
    # https://github.com/aangelopoulos/conformal-risk
    base_url = "https://raw.githubusercontent.com/aangelopoulos/conformal-risk/main/data/coco/"
    
    files = {
        "scores.npy": f"{base_url}scores.npy",
        "labels.npy": f"{base_url}labels.npy",
    }
    
    for filename, url in files.items():
        dest_path = os.path.join(data_dir, filename)
        if os.path.exists(dest_path):
            print(f"File already exists: {dest_path}")
        else:
            try:
                download_file(url, dest_path)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                print(f"Please manually download from {url}")
    
    # Verify the data
    scores_path = os.path.join(data_dir, "scores.npy")
    labels_path = os.path.join(data_dir, "labels.npy")
    
    if os.path.exists(scores_path) and os.path.exists(labels_path):
        scores = np.load(scores_path)
        labels = np.load(labels_path)
        print(f"\nData loaded successfully:")
        print(f"  Scores shape: {scores.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"  Label range: [{labels.min():.3f}, {labels.max():.3f}]")
        return scores, labels
    else:
        print("\nData download failed. Please download manually.")
        return None, None


if __name__ == "__main__":
    print("Downloading MS-COCO data...")
    scores, labels = download_coco_data("data/coco")
    if scores is not None:
        print("\nData ready for experiments!")
    else:
        print("\nPlease download data manually and place in data/coco/")
