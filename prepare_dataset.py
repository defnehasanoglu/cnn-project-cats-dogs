"""
prepare_dataset.py
──────────────────
Downloads and cleans the Microsoft Cats-vs-Dogs dataset.
Run this ONCE before train.py.

Usage:
    python prepare_dataset.py
"""

import os
import zipfile
import urllib.request
import shutil
import imghdr  # Built-in library to check file formats at the byte level
from pathlib import Path

# Apply psutil patch at the very beginning to avoid multiprocessing issues
import psutil
psutil.Process.parent = staticmethod(lambda: None)

try:
    from PIL import Image
except Exception:
    raise ImportError("Pillow is required to run this script. Install with: pip install Pillow")

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("TensorFlow is required for absolute_clean. Install with: pip install tensorflow")

DOWNLOAD_URL = (
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-"
    "6DEBA77B919F/kagglecatsanddogs_5340.zip"
)
ZIP_FILE   = "kagglecatsanddogs_5340.zip"
EXTRACT_TO = "."         # extracts → ./PetImages/
DATA_DIR   = Path("PetImages")


def download_dataset():
    if not Path(ZIP_FILE).exists():
        print("Downloading dataset (≈800 MB)…")
        urllib.request.urlretrieve(DOWNLOAD_URL, ZIP_FILE)
        print("Download complete.")
    else:
        print("Zip already present, skipping download.")


def extract_dataset():
    if DATA_DIR.exists():
        print("PetImages folder already exists, skipping extraction.")
        return
    print("Extracting…")
    with zipfile.ZipFile(ZIP_FILE) as zf:
        zf.extractall(EXTRACT_TO)
    print("Extraction complete.")


def clean_dataset():
    """
    Removes files that have corrupted data, invalid channel counts, or a .jpg 
    extension while lacking a valid image byte structure.
    (Phase 1: Pillow and imghdr based basic check)
    """
    print("━" * 60)
    print("PHASE 1: Basic corrupt and fake file scanning with Pillow started...")
    print("━" * 60)
    removed = 0
    
    # Allowed actual underlying image formats
    VALID_FORMATS = {'jpeg', 'png', 'gif', 'bmp', 'webp'}
    
    for img_path in DATA_DIR.rglob("*.jpg"):
        # Check 1: Is the file empty or completely broken?
        try:
            img = Image.open(img_path)
            img.verify() 
        except Exception:
            img_path.unlink()
            removed += 1
            continue

        # Check 2: Is the actual byte structure a valid image format?
        real_format = imghdr.what(img_path)
        if real_format not in VALID_FORMATS:
            print(f"Fake/Unknown image format found ({real_format}): {img_path}")
            img_path.unlink()  # Remove fake files that would otherwise crash TensorFlow
            removed += 1
            continue

        # Check 3: Is the channel structure standard?
        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                print(f"Bad channel image found ({img.mode}): {img_path}")
                img_path.unlink()
                removed += 1
        except Exception:
            if img_path.exists():
                img_path.unlink()
            removed += 1

    print(f"Phase 1 Complete: {removed} invalid files removed from disk.\n")


def absolute_clean():
    """
    Performs deep scanning using TensorFlow's built-in C++ functions.
    Specifically targets and removes hidden 2-channel images that can 
    break architectures like EfficientNet during training.
    (Phase 2: TensorFlow based deep cleaning)
    """
    print("━" * 60)
    print("PHASE 2: Deep Data Scanning with TensorFlow C++ Engine Started...")
    print("━" * 60)
    
    removed = 0
    # Collect all images for deep inspection
    all_images = list(DATA_DIR.rglob("*.jpg"))
    print(f"Scanning a total of {len(all_images)} files...")

    for i, img_path in enumerate(all_images):
        try:
            # 1. Read file as raw bytes
            img_bytes = tf.io.read_file(str(img_path))
            
            # 2. Attempt to decode using TensorFlow's native C++ implementation
            # Keeping channels=0 allows us to capture the original channel count
            img = tf.io.decode_image(img_bytes, channels=0, expand_animations=False)
            
            # 3. Catch files with exactly 2 channels (often incompatible with standard CNNs)
            if img.shape[-1] == 2:
                print(f"2-Channel Hidden File Found and Deleted: {img_path}")
                img_path.unlink()
                removed += 1
                
        except tf.errors.InvalidArgumentError:
            # Catch files that throw an "Unknown format" decode error in TensorFlow
            print(f"Corrupt/Fake Image Format Deleted: {img_path}")
            img_path.unlink()
            removed += 1
        except Exception as e:
            print(f"Unexpected Error (Deleting file): {img_path} -> {e}")
            if img_path.exists():
                img_path.unlink()
            removed += 1

        if (i + 1) % 5000 == 0:
            print(f"  > {i + 1} files checked...")

    print("━" * 60)
    print(f"PROCESS COMPLETED: A total of {removed} problematic files purged from disk.")
    print("━" * 60)


def print_summary():
    for cls in ["Cat", "Dog"]:
        count = len(list((DATA_DIR / cls).glob("*.jpg")))
        print(f"  {cls}: {count} images")


if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    
    # Step 1: Basic cleanup using Pillow and imghdr
    clean_dataset()
    
    # Step 2: Deep (channel-based) cleanup using TensorFlow
    absolute_clean()
    
    print("\nDataset ready:")
    print_summary()
    print("\nNext step → python train.py")