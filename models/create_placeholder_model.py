"""
models/create_placeholder_model.py
──────────────────────────────────
Creates a minimal but structurally correct CNN-LSTM .h5 model
so the Flask app can be tested end-to-end WITHOUT the full dataset.

Run once:
    python models/create_placeholder_model.py

Then train the real model using:
    python training/train_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed, Conv2D, MaxPooling2D, Flatten,
    LSTM, Dense, Dropout, BatchNormalization
)

# ── Architecture ──────────────────────────────────────────────────────────────
def build_cnn_lstm_model(input_shape=(1, 224, 224, 3), num_classes=2):
    """
    CNN-LSTM Hybrid:
      • CNN (TimeDistributed) extracts spatial features per frame
      • LSTM captures sequential / deep spatial relationships
      • Final Dense + Softmax classifies into Normal / Pneumonia
    """
    model = Sequential(name="CNN_LSTM_Pneumonia")

    # ── CNN Feature Extractor (TimeDistributed wraps each timestep) ───────────
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation="relu", padding="same"),
                              input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(64, (3, 3), activation="relu", padding="same")))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(128, (3, 3), activation="relu", padding="same")))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(256, (3, 3), activation="relu", padding="same")))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(BatchNormalization()))

    # ── Flatten each timestep's feature map ───────────────────────────────────
    model.add(TimeDistributed(Flatten()))

    # ── LSTM Layer ─────────────────────────────────────────────────────────────
    model.add(LSTM(128, return_sequences=False))

    # ── Classification Head ────────────────────────────────────────────────────
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "cnn_lstm_model.h5")

    print("Building CNN-LSTM model …")
    model = build_cnn_lstm_model()
    model.summary()

    # Save with random (untrained) weights — good enough for UI testing
    model.save(save_path)
    print(f"\n✅  Placeholder model saved to: {save_path}")
    print("   Replace this with your trained model after running train_model.py")
