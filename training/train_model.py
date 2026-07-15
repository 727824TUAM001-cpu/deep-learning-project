"""
training/train_model.py
───────────────────────
Full CNN-LSTM training script for Pneumonia Detection.

Usage:
    python training/train_model.py

Prerequisites:
    • Dataset in dataset/train/, dataset/val/, dataset/test/
      (Download from Kaggle: chest-xray-pneumonia)
    • pip install -r requirements.txt

Output:
    • models/cnn_lstm_model.h5   (best checkpoint)
    • training/training_history.npy
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.create_placeholder_model import build_cnn_lstm_model

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = os.path.join(ROOT, "dataset")
TRAIN_DIR    = os.path.join(DATASET_DIR, "train")
VAL_DIR      = os.path.join(DATASET_DIR, "val")
TEST_DIR     = os.path.join(DATASET_DIR, "test")
MODEL_SAVE   = os.path.join(ROOT, "models", "cnn_lstm_model.h5")
HISTORY_SAVE = os.path.join(ROOT, "training", "training_history.npy")

IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
EPOCHS       = 30
NUM_CLASSES  = 2


def make_generators():
    """Create augmented train and validation data generators."""

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode="nearest",
    )

    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    val_data = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_data, val_data


class LSTMAdaptorGenerator(tf.keras.utils.Sequence):
    """
    Wraps a Keras ImageDataGenerator to add the extra timestep
    dimension required by the CNN-LSTM input: (batch, 1, H, W, C).
    """

    def __init__(self, base_gen):
        self.base = base_gen

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        X, y = self.base[idx]
        return np.expand_dims(X, axis=1), y  # (B,1,H,W,C)

    def on_epoch_end(self):
        self.base.on_epoch_end()


def main():
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)

    print("=" * 60)
    print("  CNN-LSTM Pneumonia Detection — Training")
    print("=" * 60)

    # ── Dataset ───────────────────────────────────────────────────────────────
    if not os.path.isdir(TRAIN_DIR):
        print(f"\n❌  Training data not found at: {TRAIN_DIR}")
        print("    Download from Kaggle: chest-xray-pneumonia")
        sys.exit(1)

    train_data, val_data = make_generators()
    train_seq = LSTMAdaptorGenerator(train_data)
    val_seq   = LSTMAdaptorGenerator(val_data)

    print(f"\nClasses : {train_data.class_indices}")
    print(f"Train   : {train_data.samples} images")
    print(f"Val     : {val_data.samples} images")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_cnn_lstm_model(input_shape=(1, 224, 224, 3), num_classes=NUM_CLASSES)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        TensorBoard(log_dir=os.path.join(ROOT, "training", "logs")),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save history for evaluation plots
    np.save(HISTORY_SAVE, history.history)
    print(f"\n✅  Training complete. Model saved to: {MODEL_SAVE}")
    print(f"    History saved to: {HISTORY_SAVE}")


if __name__ == "__main__":
    main()
