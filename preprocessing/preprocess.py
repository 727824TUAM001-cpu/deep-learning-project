"""
preprocessing/preprocess.py
──────────────────────────
Image preprocessing utilities shared by both the training pipeline
and the Flask inference route.
"""

import os
import numpy as np
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)


def load_and_preprocess(image_path: str) -> np.ndarray:
    """
    Load an image file and return a preprocessed numpy array
    ready for CNN-LSTM inference.

    Returns shape: (1, 1, 224, 224, 3)  — batch × timestep × H × W × C
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0   # Normalize [0, 1]
    arr = np.expand_dims(arr, axis=0)               # (1, 224, 224, 3)
    arr = np.expand_dims(arr, axis=1)               # (1, 1, 224, 224, 3)
    return arr


def load_for_training(image_path: str) -> np.ndarray:
    """
    Load an image for use in Keras ImageDataGenerator pipelines.
    Returns shape: (224, 224, 3)
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def validate_image(file_path: str) -> bool:
    """Check if the file is a valid readable image."""
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception:
        return False
