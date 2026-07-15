"""
training/evaluate.py
─────────────────────
Evaluate the trained model on the test set and generate all metric
plots required for the dashboard and project report.

Usage:
    python training/evaluate.py

Output files (saved to app/static/plots/):
    • confusion_matrix.png
    • roc_curve.png
    • training_history.png
    • classification_report.txt
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

MODEL_PATH   = os.path.join(ROOT, "models", "cnn_lstm_model.h5")
TEST_DIR     = os.path.join(ROOT, "dataset", "test")
HISTORY_PATH = os.path.join(ROOT, "training", "training_history.npy")
PLOTS_DIR    = os.path.join(ROOT, "app", "static", "plots")
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
CLASS_NAMES  = ["NORMAL", "PNEUMONIA"]

os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use("dark_background")
PALETTE = ["#00b4d8", "#e63946"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model():
    print("Loading model …")
    return tf.keras.models.load_model(MODEL_PATH)


def get_test_data():
    gen = ImageDataGenerator(rescale=1.0 / 255)
    data = gen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return data


class LSTMAdaptor(tf.keras.utils.Sequence):
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        X, y = self.base[i]
        return np.expand_dims(X, axis=1), y


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PALETTE[0], lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_history(history: dict):
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["accuracy"],   color=PALETTE[0], label="Train Acc")
    ax1.plot(epochs, history["val_accuracy"], color=PALETTE[1], label="Val Acc")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(epochs, history["loss"],     color=PALETTE[0], label="Train Loss")
    ax2.plot(epochs, history["val_loss"], color=PALETTE[1], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CNN-LSTM Evaluation")
    print("=" * 60)

    model    = load_model()
    test_raw = get_test_data()
    test_seq = LSTMAdaptor(test_raw)

    print("Running predictions on test set …")
    y_prob = model.predict(test_seq, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_raw.classes

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print("\nClassification Report:\n", report)
    report_path = os.path.join(PLOTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # ── Metric summary ────────────────────────────────────────────────────────
    acc = np.mean(y_pred == y_true)
    summary = {"accuracy": round(float(acc) * 100, 2)}
    with open(os.path.join(PLOTS_DIR, "metrics.json"), "w") as f:
        json.dump(summary, f)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    print("\nGenerating plots …")
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_prob)

    if os.path.exists(HISTORY_PATH):
        history = np.load(HISTORY_PATH, allow_pickle=True).item()
        plot_training_history(history)

    print("\n✅  Evaluation complete. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
