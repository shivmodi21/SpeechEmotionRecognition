import os
import glob
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ReduceLROnPlateau,
)


from .feature_extraction_cnn import (
    extract_features,
    get_feature_config,
    FEATURE_SHAPE,
)


# ============================================================
# CONFIGURATION
# ============================================================
FEATURE_VERSION = "cnn_v3"

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "data" / "AudioWAV"
FEATURE_CACHE_DIR = BASE_DIR / "data" / "features"
MODEL_DIR = BASE_DIR / "model"
TENSORBOARD_LOG_DIR = BASE_DIR / "logs"

FEATURE_CACHE_PATH = FEATURE_CACHE_DIR / f"features_{FEATURE_VERSION}.npz"
FEATURE_CHECKPOINT_PATH = FEATURE_CACHE_DIR / f"feature_extraction_checkpoint_{FEATURE_VERSION}.npz"
SKIPPED_FILES_PATH = BASE_DIR / "data" / f"skipped_files_{FEATURE_VERSION}.json"

# Model paths
MODEL_NAME = "speech_emotion_model"
MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}_{FEATURE_VERSION}.keras"
ENCODER_PATH = MODEL_DIR / f"label_encoder_{FEATURE_VERSION}.pkl"
CONFIG_PATH = MODEL_DIR / f"feature_config_{FEATURE_VERSION}.json"
METADATA_PATH = MODEL_DIR / f"training_metadata_{FEATURE_VERSION}.json"

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 9

EPOCHS = 150
BATCH_SIZE = 32

CHECKPOINT_EVERY = 100


# ============================================================
# EMOTION MAPPING
# ============================================================

EMOTION_MAP = {
    "ANG": "angry",
    "NEU": "neutral",
    "SAD": "sad",
    "HAP": "happy",
}


# ============================================================
# LABEL / ACTOR EXTRACTION
# ============================================================

def parse_filename(file_path):
    """
    Parse a CREMA-D filename.

    Example:
        1001_DFA_ANG_XX.wav

    Returns:
        actor_id
        emotion
    """

    filename = os.path.basename(file_path)
    filename = os.path.splitext(filename)[0]

    parts = filename.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename}")

    actor_id = parts[0]
    emotion_code = parts[2]

    if emotion_code not in EMOTION_MAP:
        return None, None

    return actor_id, EMOTION_MAP[emotion_code]


# ============================================================
# FEATURE CACHE
# ============================================================

def _save_feature_checkpoint(
    features,
    labels,
    actor_ids,
    processed_files,
    total_files,
    completed=False,
    issue_records=None,
):
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        FEATURE_CHECKPOINT_PATH,
        X=np.asarray(features, dtype=np.float32),
        y=np.asarray(labels),
        actor_ids=np.asarray(actor_ids),
        processed_files=np.asarray(processed_files),
        total_files=np.int64(total_files),
        completed=np.bool_(completed),
        feature_version=FEATURE_VERSION,
        feature_shape=np.asarray(FEATURE_SHAPE, dtype=np.int64),
        issue_records_json=np.asarray(json.dumps(issue_records or [])),
    )


def _is_valid_cache(cache, files):
    """
    Check whether the completed CNN feature cache matches
    the current dataset and feature configuration.
    """

    required_keys = {
        "X",
        "y",
        "actor_ids",
        "processed_files",
        "total_files",
        "completed",
        "feature_version",
        "feature_shape",
    }

    if not required_keys.issubset(set(cache.files)):
        return False

    if not bool(cache["completed"]):
        return False

    cached_files = cache["processed_files"].tolist()

    if cached_files != files:
        return False

    if int(cache["total_files"]) != len(files):
        return False

    cached_version = str(cache["feature_version"])

    if cached_version != FEATURE_VERSION:
        return False

    cached_shape = tuple(int(value) for value in cache["feature_shape"].tolist())

    if cached_shape != tuple(FEATURE_SHAPE):
        return False

    X = cache["X"]

    if X.ndim != 4:
        return False

    if tuple(X.shape[1:]) != tuple(FEATURE_SHAPE):
        return False

    if len(cache["y"]) != len(cache["X"]):
        return False

    if len(cache["actor_ids"]) != len(cache["X"]):
        return False

    if "issue_records_json" not in cache.files:
        return False

    try:
        issue_records = json.loads(str(cache["issue_records_json"]))
    except (TypeError, json.JSONDecodeError):
        return False

    zero_feature_count = sum(
        1 for record in issue_records
        if record.get("status") == "skipped"
        and record.get("reason") == "zero_feature_tensor"
        and record.get("file") in files
    )

    if len(cache["X"]) + zero_feature_count != len(files):
        return False

    if any(record.get("status") in {"error", "missed"} and record.get("file") in files for record in issue_records):
        return False

    return True


def _load_npz(path):
    """
    Load an NPZ file and explicitly close the underlying
    NpzFile object. This avoids Windows file-lock issues
    when the cache needs to be replaced.
    """

    cache = np.load(path, allow_pickle=False)

    try:
        data = {key: cache[key] for key in cache.files}
    finally:
        cache.close()

    return data


def _write_issue_log(issue_records):
    SKIPPED_FILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SKIPPED_FILES_PATH, "w", encoding="utf-8") as file:
        json.dump(issue_records, file, indent=4)


def _record_issue(issue_records, file_path, status, reason, error_type=None, error=None):
    record = {"file": str(file_path), "status": status, "reason": reason}
    if error_type is not None:
        record["error_type"] = error_type
    if error is not None:
        record["error"] = error
    issue_records.append(record)
    _write_issue_log(issue_records)


# ============================================================
# DATASET / FEATURE EXTRACTION
# ============================================================

def load_dataset(dataset_dir):
    pattern = os.path.join(dataset_dir, "**", "*.wav")
    all_files = sorted(glob.glob(pattern, recursive=True))
    if not all_files:
        raise FileNotFoundError(f"No WAV files found in: {dataset_dir}")

    issue_records = []
    files = []

    for file_path in all_files:
        try:
            actor_id, emotion = parse_filename(file_path)
        except Exception as error:
            _record_issue(issue_records, file_path, "error", "filename_parse_failed", type(error).__name__, repr(error))
            continue
        if emotion is None:
            _record_issue(issue_records, file_path, "skipped", "unsupported_emotion")
            continue
        files.append(file_path)

    print(f"Found {len(all_files)} WAV files.")
    print(f"Training candidates: {len(files)}")
    print(f"Skipped during dataset scan: {len(issue_records)}")
    if not files:
        _write_issue_log(issue_records)
        raise FileNotFoundError("No supported WAV files found for training.")

    if FEATURE_CACHE_PATH.exists():
        cache = _load_npz(FEATURE_CACHE_PATH)
        if _is_valid_cache(cache, files):
            X = cache["X"].astype(np.float32, copy=False)
            y = cache["y"]
            actor_ids = cache["actor_ids"]
            _write_issue_log(issue_records)
            print(f"\nLoaded cached CNN features: {FEATURE_CACHE_PATH}")
            print(f"Feature version: {FEATURE_VERSION}")
            print(f"Feature shape: {X.shape}")
            print(f"Labels shape: {y.shape}")
            print(f"Actor IDs: {len(np.unique(actor_ids))} unique actors")
            return X, y, actor_ids
        print("\nExisting CNN feature cache is incompatible with the current dataset or feature configuration.")
        print("Rebuilding CNN feature cache...")
        FEATURE_CACHE_PATH.unlink()

    features, labels, actor_ids, processed_files = [], [], [], []

    if FEATURE_CHECKPOINT_PATH.exists():
        cache = _load_npz(FEATURE_CHECKPOINT_PATH)
        try:
            required_keys = {"X", "y", "actor_ids", "processed_files", "total_files", "feature_version", "feature_shape"}
            metadata_valid = (
                required_keys.issubset(set(cache.keys()))
                and str(cache["feature_version"]) == FEATURE_VERSION
                and tuple(int(value) for value in cache["feature_shape"].tolist()) == tuple(FEATURE_SHAPE)
            )
            cached_files = cache["processed_files"].tolist()
            dataset_valid = int(cache["total_files"]) == len(files) and set(cached_files).issubset(set(files))
            shape_valid = (
                cache["X"].ndim == 4
                and tuple(cache["X"].shape[1:]) == tuple(FEATURE_SHAPE)
                and len(cache["X"]) == len(cache["y"]) == len(cache["actor_ids"]) == len(cached_files)
            )
            if metadata_valid and dataset_valid and shape_valid:
                features = list(cache["X"])
                labels = list(cache["y"])
                actor_ids = list(cache["actor_ids"])
                processed_files = cached_files
                if "issue_records_json" in cache:
                    try:
                        issue_records = json.loads(str(cache["issue_records_json"]))
                    except (TypeError, json.JSONDecodeError):
                        pass
                print(f"\nResuming CNN feature extraction: {len(processed_files)}/{len(files)} files already processed.")
            else:
                print("\nExisting CNN feature checkpoint is incompatible. Starting fresh.")
                FEATURE_CHECKPOINT_PATH.unlink()
        finally:
            pass

    processed_set = set(processed_files)

    for index, file_path in enumerate(files):
        if file_path in processed_set:
            continue
        try:
            actor_id, emotion = parse_filename(file_path)
            feature_tensor = extract_features(file_path)
            if feature_tensor.shape != FEATURE_SHAPE:
                raise ValueError(f"Expected feature shape {FEATURE_SHAPE}, got {feature_tensor.shape}")

            if not np.any(feature_tensor):
                print(f"Skipping zero-feature file: {file_path}")
                _record_issue(issue_records, file_path, "skipped", "zero_feature_tensor")
                processed_files.append(file_path)
                processed_set.add(file_path)
            else:
                features.append(feature_tensor)
                labels.append(emotion)
                actor_ids.append(actor_id)
                processed_files.append(file_path)
                processed_set.add(file_path)
        except Exception as error:
            print(f"\nFailed: {file_path}\nError Type: {type(error).__name__}\nReason: {repr(error)}")
            _record_issue(issue_records, file_path, "error", "feature_extraction_failed", type(error).__name__, repr(error))

        current_count = index + 1
        if current_count % CHECKPOINT_EVERY == 0:
            _save_feature_checkpoint(features, labels, actor_ids, processed_files, len(files), completed=False, issue_records=issue_records)
            print(f"Processed {current_count}/{len(files)} dataset positions - checkpoint saved.")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    actor_ids = np.asarray(actor_ids)

    print("\nDataset loaded.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Actor IDs shape: {actor_ids.shape}")

    missing_training_files = [file_path for file_path in files if file_path not in processed_set]
    if missing_training_files:
        for file_path in missing_training_files:
            _record_issue(issue_records, file_path, "missed", "not_processed")
        _save_feature_checkpoint(features, labels, actor_ids, processed_files, len(files), completed=False, issue_records=issue_records)
        raise RuntimeError(f"Feature extraction did not complete for {len(missing_training_files)} training files. See {SKIPPED_FILES_PATH}.")

    if issue_records:
        print(f"\nWARNING: {len(issue_records)} files were skipped or had errors.")
        _write_issue_log(issue_records)

    if len(X) == 0:
        raise RuntimeError("No valid feature tensors were extracted for training.")

    if len(X) != len(y) or len(X) != len(actor_ids):
        raise RuntimeError("Feature, label, and actor ID counts do not match.")

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for emotion, count in zip(unique, counts):
        print(f"{emotion:10s}: {count}")

    _save_feature_checkpoint(features, labels, actor_ids, processed_files, len(files), completed=True, issue_records=issue_records)
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FEATURE_CACHE_PATH,
        X=X,
        y=y,
        actor_ids=actor_ids,
        processed_files=np.asarray(processed_files),
        total_files=np.int64(len(files)),
        completed=np.bool_(True),
        feature_version=FEATURE_VERSION,
        feature_shape=np.asarray(FEATURE_SHAPE, dtype=np.int64),
        issue_records_json=np.asarray(json.dumps(issue_records)),
    )
    print(f"\nFinal CNN feature cache saved: {FEATURE_CACHE_PATH}")
    if FEATURE_CHECKPOINT_PATH.exists():
        FEATURE_CHECKPOINT_PATH.unlink()
    _write_issue_log(issue_records)
    return X, y, actor_ids

# ============================================================
# ACTOR-AWARE SPLIT
# ============================================================

def create_actor_aware_split(
    actor_ids,
    test_size=TEST_SIZE,
    validation_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
):
    """
    Split actors first, then return sample indices.

    This guarantees that no actor appears in more than
    one of train, validation, and test sets.
    """

    unique_actors = np.unique(actor_ids)

    if len(unique_actors) < 3:
        raise ValueError("At least 3 unique actors are required for an actor-aware train/validation/test split.")

    train_val_actors, test_actors = train_test_split(unique_actors, test_size=test_size, random_state=random_state)
    validation_relative_size = validation_size / (1.0 - test_size)
    train_actors, validation_actors = train_test_split(train_val_actors, test_size=validation_relative_size, random_state=random_state)
    train_actors = set(train_actors.tolist())
    validation_actors = set(validation_actors.tolist())
    test_actors = set(test_actors.tolist())
    train_indices = np.array([index for index, actor in enumerate(actor_ids) if actor in train_actors], dtype=np.int64)
    validation_indices = np.array([index for index, actor in enumerate(actor_ids) if actor in validation_actors], dtype=np.int64)
    test_indices = np.array([index for index, actor in enumerate(actor_ids) if actor in test_actors], dtype=np.int64)

    # Verify there is no actor leakage
    assert train_actors.isdisjoint(validation_actors)
    assert train_actors.isdisjoint(test_actors)
    assert validation_actors.isdisjoint(test_actors)

    return train_indices, validation_indices, test_indices, sorted(train_actors), sorted(validation_actors), sorted(test_actors)


# ============================================================
# BUILD CNN
# ============================================================

def build_model(input_shape, num_classes,):
    """
    CNN for log-Mel spectrogram classification.
    """

    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            GlobalAveragePooling2D(),
            Dense(128, activation="relu"),
            Dropout(0.4),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y, actor_ids = load_dataset(DATASET_DIR)

    print("\nFinal feature shape:")
    print(X.shape)

    train_indices, validation_indices, test_indices, train_actors, validation_actors, test_actors = create_actor_aware_split(actor_ids)

    X_train = X[train_indices]
    X_validation = X[validation_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_validation = y[validation_indices]
    y_test = y[test_indices]

    print("\n==============================")
    print("ACTOR-AWARE SPLIT")
    print("==============================")

    print(f"Training actors:   {len(train_actors)}")
    print(f"Validation actors: {len(validation_actors)}")
    print(f"Testing actors:    {len(test_actors)}")
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")
    print(f"Testing samples:    {len(X_test)}")

    print("\nActor overlap check:")
    print(f"Train ∩ Validation: {len(set(train_actors) & set(validation_actors))}")
    print(f"Train ∩ Test: {len(set(train_actors) & set(test_actors))}")
    print(f"Validation ∩ Test: {len(set(validation_actors) & set(test_actors))}")

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_validation_encoded = label_encoder.transform(y_validation)
    y_test_encoded = label_encoder.transform(y_test)

    num_classes = len(label_encoder.classes_)

    expected_classes = set(EMOTION_MAP.values())
    available_classes = set(np.unique(y_train))
    missing_classes = sorted(expected_classes - available_classes)
    if missing_classes:
        raise RuntimeError(
            "Training dataset is missing supported emotion classes: " + ", ".join(missing_classes)
        )

    print("\nClasses:")
    for index, emotion in enumerate(label_encoder.classes_):
        print(f"{index}: {emotion}")

    model = build_model(input_shape=FEATURE_SHAPE, num_classes=num_classes)
    model.summary()

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_log_dir = TENSORBOARD_LOG_DIR / run_name
    tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor="val_accuracy", mode="max", patience=20, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=7, min_lr=1e-6, verbose=1)

    print("\n==============================")
    print("TRAINING CNN")
    print("==============================")

    history = model.fit(
        X_train,
        y_train_encoded,
        validation_data=(X_validation, y_validation_encoded),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=[tensorboard_callback, reduce_lr, early_stopping],
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print("\nClassification Report:\n")
    print(classification_report(y_test_encoded, predictions, target_names=label_encoder.classes_))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test_encoded, predictions))

    model.save(MODEL_PATH)

    with open(ENCODER_PATH, "wb") as file:
        pickle.dump(label_encoder, file)

    # --------------------------------------------------------
    # Save CNN feature configuration
    # --------------------------------------------------------

    feature_config = get_feature_config()

    feature_config.update(
        {
            "emotions": label_encoder.classes_.tolist(),
            "validation_size": VALIDATION_SIZE,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "split_strategy": "actor_aware",
            "train_actor_count": len(train_actors),
            "validation_actor_count": len(validation_actors),
            "test_actor_count": len(test_actors),
            "model_type": "2d_cnn",
            "model_input_shape": list(FEATURE_SHAPE),
        }
    )

    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(feature_config, file, indent=4)

    training_metadata = {
        "feature_version": FEATURE_VERSION,
        "feature_shape": list(FEATURE_SHAPE),
        "model_type": "2d_cnn",
        "dataset": "CREMA-D",
        "dataset_path": str(DATASET_DIR),
        "random_state": RANDOM_STATE,
        "split_strategy": "actor_aware",
        "train_samples": len(X_train),
        "validation_samples": len(X_validation),
        "test_samples": len(X_test),
        "train_actors": train_actors,
        "validation_actors": validation_actors,
        "test_actors": test_actors,
        "epochs_requested": EPOCHS,
        "batch_size": BATCH_SIZE,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "training_completed_at": datetime.now().isoformat(),
        "tensorboard_log_dir": str(tensorboard_log_dir),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(training_metadata, file, indent=4)

    print(f"\nTensorBoard logs: {tensorboard_log_dir}")
    print("\nModel artifacts saved:")
    print(f"  {MODEL_PATH}")
    print(f"  {ENCODER_PATH}")
    print(f"  {CONFIG_PATH}")
    print(f"  {METADATA_PATH}")

if __name__ == "__main__":
    main()
