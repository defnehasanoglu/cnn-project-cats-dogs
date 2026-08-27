"""
make_mini_dataset.py
─────────────────────
Creates a tiny fake "PetImages" folder with a handful of random JPEGs
per class, just so train.py can run end-to-end locally in under a minute
to catch bugs (shape errors, typos, broken callbacks, etc.) before you
burn HPC budget on the real 25,000-image dataset.

This does NOT test model accuracy (random noise images learn nothing
meaningful) — it only tests that the pipeline runs without crashing.

Usage:
    python make_mini_dataset.py
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image

OUT_DIR = Path("PetImages_mini")
IMAGES_PER_CLASS = 40       
IMG_SIZE = (224, 224)

def make_class(name: str, n: int):
    class_dir = OUT_DIR / name
    class_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42 if name == "Cat" else 43)
    for i in range(n):
        # Random noise image — fine for a pipeline smoke test
        arr = rng.integers(0, 255, size=(*IMG_SIZE, 3), dtype=np.uint8)
        Image.fromarray(arr).save(class_dir / f"{i}.jpg", quality=90)

if __name__ == "__main__":
    print(f"Creating mini dataset at {OUT_DIR}/ ...")
    make_class("Cat", IMAGES_PER_CLASS)
    make_class("Dog", IMAGES_PER_CLASS)
    print(f"Done: {IMAGES_PER_CLASS} Cat + {IMAGES_PER_CLASS} Dog images.")
    print(f"\nNext: python train.py --smoke-test")
