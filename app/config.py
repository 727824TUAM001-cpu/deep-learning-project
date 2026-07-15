import os
from dotenv import load_dotenv

# Load .env from project root (two levels up from app/config.py)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(_ROOT, ".env"))

BASE_DIR = os.path.abspath(os.path.dirname(__file__))   # .../app/
ROOT_DIR = _ROOT                                         # project root


class Config:
    # ── Security ──────────────────────────────────────────────────────────────
    SECRET_KEY = os.environ.get("SECRET_KEY", "pneumonia-ai-secret-2024")

    # ── Database ──────────────────────────────────────────────────────────────
    # SQLite (default — no installation required)
    _db_url = os.environ.get("DATABASE_URL",
                             f"sqlite:///{os.path.join(ROOT_DIR, 'database', 'pneumonia.db')}")
    # Fix relative sqlite:/// paths so they resolve from project root
    SQLALCHEMY_DATABASE_URI = _db_url.replace(
        "sqlite:///database/", f"sqlite:///{os.path.join(ROOT_DIR, 'database', '')}"
    ) if _db_url.startswith("sqlite:///database/") else _db_url
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ── File uploads ──────────────────────────────────────────────────────────
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

    # ── Model ─────────────────────────────────────────────────────────────────
    MODEL_PATH = os.path.join(ROOT_DIR, "models", "cnn_lstm_model.h5")

    # ── Plots ─────────────────────────────────────────────────────────────────
    PLOTS_FOLDER = os.path.join(BASE_DIR, "static", "plots")
