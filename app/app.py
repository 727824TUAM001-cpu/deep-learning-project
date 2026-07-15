"""
app/app.py
──────────
Flask application entry point for AI-Pneumonia Detection.

Routes:
  /              → redirect to login
  /register      → user registration
  /login         → login
  /logout        → logout
  /dashboard     → stats + charts
  /predict       → upload X-ray + run model
  /history       → prediction history
  /admin         → admin panel
  /admin/delete_user/<id>  → delete user (admin only)
  /api/stats     → JSON stats for Chart.js
"""

import os
import sys
import uuid
import json
from datetime import datetime, timedelta

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify, send_from_directory)
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, current_user)
from werkzeug.utils import secure_filename
from sqlalchemy import func

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR  = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from app.config import Config
from app.models_db import db, User, Prediction

# ── Lazy TF import (avoids slow startup on import) ───────────────────────────
_model = None


def get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        model_path = Config.MODEL_PATH
        if os.path.exists(model_path):
            _model = tf.keras.models.load_model(model_path)
            print(f"[OK] Model loaded from {model_path}")
        else:
            print(f"[WARN] Model not found at {model_path}")
            print("       Run: python models/create_placeholder_model.py")
    return _model


# ── Factory ───────────────────────────────────────────────────────────────────
def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Ensure directories exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["PLOTS_FOLDER"], exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "database"), exist_ok=True)

    # ── Extensions ────────────────────────────────────────────────────────────
    db.init_app(app)

    login_manager = LoginManager(app)
    login_manager.login_view = "login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "info"

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # ── DB init + seed ────────────────────────────────────────────────────────
    with app.app_context():
        db.create_all()
        _seed_admin()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def allowed_file(filename):
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
        )

    def run_inference(image_path: str) -> dict:
        """Run CNN-LSTM model on an image and return prediction dict."""
        import numpy as np
        from preprocessing.preprocess import load_and_preprocess

        arr = load_and_preprocess(image_path)
        model = get_model()

        if model is None:
            # Fallback mock when model is missing (dev mode)
            import random
            pneumonia_prob = random.uniform(0.45, 0.99)
            normal_prob    = 1.0 - pneumonia_prob
            prediction     = "Pneumonia" if pneumonia_prob >= 0.5 else "Normal"
            confidence     = max(pneumonia_prob, normal_prob) * 100
        else:
            probs          = model.predict(arr, verbose=0)[0]
            normal_prob    = float(probs[0])
            pneumonia_prob = float(probs[1])
            prediction     = "Pneumonia" if pneumonia_prob >= 0.5 else "Normal"
            confidence     = max(normal_prob, pneumonia_prob) * 100

        return {
            "prediction":     prediction,
            "confidence":     round(confidence, 2),
            "normal_prob":    round(normal_prob * 100, 2),
            "pneumonia_prob": round(pneumonia_prob * 100, 2),
        }

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return redirect(url_for("dashboard" if current_user.is_authenticated else "login"))

    # ── Auth ──────────────────────────────────────────────────────────────────

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            email    = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm  = request.form.get("confirm_password", "")

            if not all([username, email, password, confirm]):
                flash("All fields are required.", "danger")
            elif password != confirm:
                flash("Passwords do not match.", "danger")
            elif len(password) < 8:
                flash("Password must be at least 8 characters.", "danger")
            elif User.query.filter_by(email=email).first():
                flash("Email already registered.", "danger")
            elif User.query.filter_by(username=username).first():
                flash("Username already taken.", "danger")
            else:
                user = User(username=username, email=email)
                user.set_password(password)
                db.session.add(user)
                db.session.commit()
                flash("Registration successful! Please log in.", "success")
                return redirect(url_for("login"))

        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        if request.method == "POST":
            email    = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            user     = User.query.filter_by(email=email).first()

            if user and user.check_password(password):
                login_user(user, remember=request.form.get("remember"))
                next_page = request.args.get("next")
                flash(f"Welcome back, {user.username}!", "success")
                return redirect(next_page or url_for("dashboard"))
            else:
                flash("Invalid email or password.", "danger")

        return render_template("login.html")

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        flash("You have been logged out.", "info")
        return redirect(url_for("login"))

    # ── Dashboard ─────────────────────────────────────────────────────────────

    @app.route("/dashboard")
    @login_required
    def dashboard():
        scope = Prediction.query if current_user.is_admin else \
                Prediction.query.filter_by(user_id=current_user.id)

        total      = scope.count()
        pneumonia  = scope.filter_by(prediction="Pneumonia").count()
        normal     = scope.filter_by(prediction="Normal").count()
        recent     = scope.order_by(Prediction.prediction_date.desc()).limit(5).all()

        # Accuracy from metrics.json if available
        metrics_path = os.path.join(app.config["PLOTS_FOLDER"], "metrics.json")
        model_accuracy = None
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                model_accuracy = json.load(f).get("accuracy")

        return render_template(
            "dashboard.html",
            total=total, pneumonia=pneumonia, normal=normal,
            recent=recent, model_accuracy=model_accuracy,
        )

    # ── Predict ───────────────────────────────────────────────────────────────

    @app.route("/predict", methods=["GET", "POST"])
    @login_required
    def predict():
        result = None
        if request.method == "POST":
            patient_name = request.form.get("patient_name", "Unknown").strip()
            age          = request.form.get("age", None)
            gender       = request.form.get("gender", "")
            file         = request.files.get("xray_image")

            if not file or file.filename == "":
                flash("Please select an X-ray image.", "danger")
                return redirect(request.url)

            if not allowed_file(file.filename):
                flash("Invalid file type. Only JPG, JPEG, PNG allowed.", "danger")
                return redirect(request.url)

            # Save file with unique name
            ext      = file.filename.rsplit(".", 1)[1].lower()
            filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Run inference
            result = run_inference(filepath)

            # Save to database
            pred = Prediction(
                user_id=current_user.id,
                patient_name=patient_name,
                age=int(age) if age and age.isdigit() else None,
                gender=gender,
                image_path=f"uploads/{filename}",
                prediction=result["prediction"],
                confidence=result["confidence"],
                normal_prob=result["normal_prob"],
                pneumonia_prob=result["pneumonia_prob"],
            )
            db.session.add(pred)
            db.session.commit()
            result["id"] = pred.id

        return render_template("predict.html", result=result)

    # ── History ───────────────────────────────────────────────────────────────

    @app.route("/history")
    @login_required
    def history():
        scope = Prediction.query if current_user.is_admin else \
                Prediction.query.filter_by(user_id=current_user.id)
        predictions = scope.order_by(Prediction.prediction_date.desc()).all()
        return render_template("history.html", predictions=predictions)

    # ── Admin ─────────────────────────────────────────────────────────────────

    @app.route("/admin")
    @login_required
    def admin():
        if not current_user.is_admin:
            flash("Access denied. Admins only.", "danger")
            return redirect(url_for("dashboard"))
        users       = User.query.order_by(User.created_at.desc()).all()
        total_preds = Prediction.query.count()
        total_users = User.query.count()
        return render_template(
            "admin.html",
            users=users,
            total_preds=total_preds,
            total_users=total_users,
        )

    @app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
    @login_required
    def delete_user(user_id):
        if not current_user.is_admin:
            return jsonify({"error": "Forbidden"}), 403
        if user_id == current_user.id:
            flash("You cannot delete your own account.", "danger")
            return redirect(url_for("admin"))
        user = User.query.get_or_404(user_id)
        Prediction.query.filter_by(user_id=user_id).delete()
        db.session.delete(user)
        db.session.commit()
        flash(f"User '{user.username}' deleted.", "success")
        return redirect(url_for("admin"))

    # ── API ───────────────────────────────────────────────────────────────────

    @app.route("/api/stats")
    @login_required
    def api_stats():
        """JSON endpoint for Chart.js — predictions over last 7 days."""
        today  = datetime.utcnow().date()
        labels = []
        normal_counts     = []
        pneumonia_counts  = []

        scope = Prediction.query if current_user.is_admin else \
                Prediction.query.filter_by(user_id=current_user.id)

        for i in range(6, -1, -1):
            day   = today - timedelta(days=i)
            start = datetime.combine(day, datetime.min.time())
            end   = datetime.combine(day, datetime.max.time())

            n = scope.filter(
                Prediction.prediction == "Normal",
                Prediction.prediction_date.between(start, end),
            ).count()
            p = scope.filter(
                Prediction.prediction == "Pneumonia",
                Prediction.prediction_date.between(start, end),
            ).count()

            labels.append(day.strftime("%b %d"))
            normal_counts.append(n)
            pneumonia_counts.append(p)

        total_n = scope.filter_by(prediction="Normal").count()
        total_p = scope.filter_by(prediction="Pneumonia").count()

        return jsonify({
            "labels": labels,
            "normal": normal_counts,
            "pneumonia": pneumonia_counts,
            "donut": {"normal": total_n, "pneumonia": total_p},
        })

    @app.route("/api/delete_prediction/<int:pred_id>", methods=["POST"])
    @login_required
    def delete_prediction(pred_id):
        pred = Prediction.query.get_or_404(pred_id)
        if pred.user_id != current_user.id and not current_user.is_admin:
            return jsonify({"error": "Forbidden"}), 403
        db.session.delete(pred)
        db.session.commit()
        return jsonify({"success": True})

    # ── Static uploads ────────────────────────────────────────────────────────
    @app.route("/uploads/<path:filename>")
    @login_required
    def uploaded_file(filename):
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    return app


# ── Seed admin ────────────────────────────────────────────────────────────────

def _seed_admin():
    """Create default admin user if not present."""
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@pneumonia.ai")
    if not User.query.filter_by(email=admin_email).first():
        admin = User(
            username=os.environ.get("ADMIN_USERNAME", "admin"),
            email=admin_email,
            role="admin",
        )
        admin.set_password(os.environ.get("ADMIN_PASSWORD", "Admin@123"))
        db.session.add(admin)
        db.session.commit()
        print(f"[OK] Admin user created: {admin_email}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT_DIR, ".env"))

    app = create_app()
    print("\n" + "=" * 60)
    print("  AI Pneumonia Detection - Flask Server")
    print("  http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
