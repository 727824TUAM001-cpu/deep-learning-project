from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """Registered users (doctors, radiologists, admin)."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default="user")  # "user" | "admin"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship
    predictions = db.relationship("Prediction", backref="user", lazy=True)

    # ── Password helpers ──────────────────────────────────────────────────────
    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def __repr__(self):
        return f"<User {self.username}>"


class Prediction(db.Model):
    """Stores every X-ray prediction result."""

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    patient_name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    image_path = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)  # "Normal" | "Pneumonia"
    confidence = db.Column(db.Float, nullable=False)
    normal_prob = db.Column(db.Float, nullable=False)
    pneumonia_prob = db.Column(db.Float, nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "patient_name": self.patient_name,
            "age": self.age,
            "gender": self.gender,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 2),
            "normal_prob": round(self.normal_prob, 2),
            "pneumonia_prob": round(self.pneumonia_prob, 2),
            "date": self.prediction_date.strftime("%Y-%m-%d %H:%M"),
            "image_path": self.image_path,
        }

    def __repr__(self):
        return f"<Prediction {self.patient_name} → {self.prediction}>"
