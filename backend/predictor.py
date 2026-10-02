import json
import pickle
from pathlib import Path
import threading

import numpy as np
from tensorflow.keras.models import load_model

from .feature_extraction import extract_features

FEATURE_VERSION = "v3"
MODEL_NAME = "speech_emotion_model"

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"

# Model paths
MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}_{FEATURE_VERSION}.keras"
SCALER_PATH = MODEL_DIR / f"scaler_{FEATURE_VERSION}.pkl"
ENCODER_PATH = MODEL_DIR / f"label_encoder_{FEATURE_VERSION}.pkl"

_model = None
_scaler = None
_labels = None

_artifacts_lock = threading.Lock()


def load_artifacts():
    global _model, _scaler, _labels
    if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not ENCODER_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Run backend/train_model.py first."
        )

    with _artifacts_lock:
        if _model is None:
            _model = load_model(MODEL_PATH)
        if _scaler is None:
            with open(SCALER_PATH, "rb") as f:
                _scaler = pickle.load(f)
        if _labels is None:
            with open(ENCODER_PATH, "rb") as f:
                _labels = pickle.load(f)
    
    return _model, _scaler, _labels

def predict_emotion(file_path: str):
    model, scaler, labels = load_artifacts()

    if model.output_shape[-1] != len(labels.classes_):
        raise ValueError(
            "Model output classes do not match the label encoder."
        )

    features = extract_features(file_path)
    scaled = scaler.transform(np.asarray([features], dtype=np.float32))
    probabilities = model.predict(scaled, verbose=0)[0]
    index = int(np.argmax(probabilities))
    emotion = str(labels.inverse_transform([index])[0])
    confidence = float(probabilities[index])

    return {
        "emotion": emotion,
        "confidence": confidence,
        "probabilities": {
            str(label): float(probabilities[i])
            for i, label in enumerate(labels.classes_)
        },
    }
