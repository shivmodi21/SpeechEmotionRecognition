import json
import pickle
from pathlib import Path
import threading

import numpy as np
from tensorflow.keras.models import load_model

# from src.feature_extraction_acoustic import extract_features as extract_acoustic_features
from src.feature_extraction_cnn import extract_features as extract_cnn_features

FEATURE_VERSION = "cnn_v1"
MODEL_NAME = "speech_emotion_model"

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"
HELPER_DIR = MODEL_DIR / "helper"

# Model paths
MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}_{FEATURE_VERSION}.keras"
ENCODER_PATH = HELPER_DIR / f"label_encoder_{FEATURE_VERSION}.pkl"

_model = None
_labels = None

_artifacts_lock = threading.Lock()


def load_artifacts():
    global _model, _labels
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        raise FileNotFoundError("Model artifacts are missing. Run backend/train_model.py first")

    with _artifacts_lock:
        if _model is None:
            _model = load_model(MODEL_PATH)
        if _labels is None:
            with open(ENCODER_PATH, "rb") as f:
                _labels = pickle.load(f)

    return _model, _labels

def predict_emotion(file_path: str):
    model, labels = load_artifacts()

    if model.output_shape[-1] != len(labels.classes_):
        raise ValueError("Model output classes do not match the label encoder")

    features = extract_cnn_features(file_path)
    probabilities = model.predict(np.asarray([features], dtype=np.float32), verbose=0)[0]
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
