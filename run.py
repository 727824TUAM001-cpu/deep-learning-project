"""
run.py — top-level entry point (run from project root)

Usage:
    python run.py
"""
import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from app.app import create_app

app = create_app()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PneumoAI - Flask Dev Server")
    print("  http://127.0.0.1:5000")
    print("  Admin: admin@pneumonia.ai / Admin@123")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
