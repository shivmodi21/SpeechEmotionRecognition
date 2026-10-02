import os
import glob
import json
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

from .feature_extraction import extract_features


# ============================================================
# CONFIGURATION
# ============================================================

# Feature extraction version
FEATURE_VERSION = "v3"

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
SCALER_PATH = MODEL_DIR / f"scaler_{FEATURE_VERSION}.pkl"
ENCODER_PATH = MODEL_DIR / f"label_encoder_{FEATURE_VERSION}.pkl"
CONFIG_PATH = MODEL_DIR / f"feature_config_{FEATURE_VERSION}.json"
METADATA_PATH = MODEL_DIR / f"training_metadata_{FEATURE_VERSION}.json"

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
TRAIN_SIZE = 1.0 - VALIDATION_SIZE - TEST_SIZE
RANDOM_STATE = 9
EPOCHS = 150
BATCH_SIZE = 10
CHECKPOINT_EVERY = 100
EXPECTED_FEATURE_COUNT = 550

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
# FILENAME PARSING
# ============================================================

def parse_filename(file_path):
    """
    Parse a CREMA-D filename.

    Example:

        1001_DFA_ANG_XX.wav

    Parts:

        1001 -> Actor ID
        DFA  -> Sentence
        ANG  -> Emotion
        XX   -> Intensity
    """

    filename = os.path.basename(file_path).split(".")[0]
    parts = filename.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename}")

    actor_id = parts[0]
    emotion_code = parts[2]

    if emotion_code not in EMOTION_MAP:
        return None, None

    return actor_id, EMOTION_MAP[emotion_code]

def get_emotion_from_filename(file_path):
    _, emotion = parse_filename(file_path)
    return emotion

def get_actor_id_from_filename(file_path):
    actor_id, _ = parse_filename(file_path)
    return actor_id


# ============================================================
# FEATURE CHECKPOINT
# ============================================================

def _save_feature_checkpoint(features, labels, actor_ids, processed_files, total_files, completed=False, issue_records=None):
    """Save the current feature extraction state.

    Actor IDs are stored alongside features and labels so the
    completed cache and extraction checkpoint remain usable
    for actor-aware splitting."""

    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        FEATURE_CHECKPOINT_PATH,
        X=np.asarray(features, dtype=np.float32),
        y=np.asarray(labels),
        actor_ids=np.asarray(actor_ids),
        processed_files=np.asarray(processed_files),
        total_files=np.int64(total_files),
        completed=np.bool_(completed),
        feature_count=np.int64(EXPECTED_FEATURE_COUNT),
        feature_version=FEATURE_VERSION,
        issue_records_json=np.asarray(json.dumps(issue_records or [])),
    )

def _write_issue_log(issue_records):
    SKIPPED_FILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SKIPPED_FILES_PATH, "w", encoding="utf-8") as file:
        json.dump(issue_records, file, indent=4)


def _record_issue(issue_records, file_path, status, reason, error_type=None, error=None):
    record = {
        "file": str(file_path),
        "status": status,
        "reason": reason,
    }
    if error_type is not None:
        record["error_type"] = error_type
    if error is not None:
        record["error"] = error
    issue_records.append(record)
    _write_issue_log(issue_records)


# ============================================================
# FEATURE CACHE VALIDATION
# ============================================================

def _is_valid_cache(cache, files):
    """
    Check whether a completed feature cache matches:

    1. Current dataset files
    2. Current feature version
    3. Current feature count
    4. Actor ID metadata
    """

    required_keys = {
        "X",
        "y",
        "actor_ids",
        "processed_files",
        "total_files",
        "completed",
        "feature_count",
        "feature_version",
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

    if int(cache["feature_count"]) != EXPECTED_FEATURE_COUNT:
        return False

    if str(cache["feature_version"]) != FEATURE_VERSION:
        return False

    X = cache["X"]
    y = cache["y"]
    actor_ids = cache["actor_ids"]

    if X.ndim != 2:
        return False

    if X.shape[0] != len(files):
        return False

    if X.shape[1] != EXPECTED_FEATURE_COUNT:
        return False

    if len(y) != len(X):
        return False

    if actor_ids.ndim != 1:
        return False

    if len(actor_ids) != len(X):
        return False

    return True


# ============================================================
# DATASET LOADING / FEATURE EXTRACTION
# ============================================================

def load_dataset(dataset_dir):
    pattern = os.path.join(dataset_dir, "**", "*.wav")
    all_files = sorted(glob.glob(pattern, recursive=True))

    if not all_files:
        raise FileNotFoundError(f"No WAV files found in: {dataset_dir}")

    issue_records = []
    files = []

    # Keep every WAV visible to the pipeline, but only train on the four
    # supported emotions. Unsupported CREMA-D emotions are intentional skips.
    for file_path in all_files:
        try:
            actor_id, emotion = parse_filename(file_path)
        except Exception as error:
            _record_issue(
                issue_records,
                file_path,
                "error",
                "filename_parse_failed",
                type(error).__name__,
                repr(error),
            )
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
        cache = np.load(FEATURE_CACHE_PATH, allow_pickle=False)
        try:
            cache_is_valid = _is_valid_cache(cache, files)
            if cache_is_valid:
                X = cache["X"].astype(np.float32, copy=False)
                y = cache["y"].copy()
                actor_ids = cache["actor_ids"].copy()
                print(f"\nLoaded cached features: {FEATURE_CACHE_PATH}")
                print(f"Feature version: {FEATURE_VERSION}")
                print(f"Feature count: {X.shape[1]}")
                print(f"X shape: {X.shape}")
                print(f"y shape: {y.shape}")
                print(f"Actor IDs: {len(np.unique(actor_ids))} unique actors")
                _write_issue_log(issue_records)
                return X, y, actor_ids
        finally:
            cache.close()

        print("\nExisting feature cache is incompatible with the current feature configuration or dataset.")
        print("Rebuilding feature cache from scratch...")
        FEATURE_CACHE_PATH.unlink()

    features = []
    labels = []
    actor_ids = []
    processed_files = []

    if FEATURE_CHECKPOINT_PATH.exists():
        cache = np.load(FEATURE_CHECKPOINT_PATH, allow_pickle=False)
        try:
            required_keys = {"X", "y", "actor_ids", "processed_files", "total_files", "feature_count", "feature_version"}
            checkpoint_metadata_valid = (
                required_keys.issubset(set(cache.files))
                and int(cache["feature_count"]) == EXPECTED_FEATURE_COUNT
                and str(cache["feature_version"]) == FEATURE_VERSION
            )
            cached_files = cache["processed_files"].tolist()
            cached_total = int(cache["total_files"])
            checkpoint_dataset_valid = cached_total == len(files) and set(cached_files).issubset(set(files))
            checkpoint_shape_valid = (
                cache["X"].ndim == 2
                and cache["X"].shape[1] == EXPECTED_FEATURE_COUNT
                and len(cache["X"]) == len(cache["y"]) == len(cache["actor_ids"]) == len(cached_files)
            )
            if checkpoint_metadata_valid and checkpoint_dataset_valid and checkpoint_shape_valid:
                features = list(cache["X"])
                labels = list(cache["y"])
                actor_ids = list(cache["actor_ids"])
                processed_files = cached_files
                if "issue_records_json" in cache.files:
                    try:
                        issue_records = json.loads(str(cache["issue_records_json"]))
                    except (TypeError, json.JSONDecodeError):
                        pass
                print(f"\nResuming feature extraction: {len(processed_files)}/{len(files)} files already processed.")
            else:
                print("\nExisting feature checkpoint is incompatible. Starting fresh.")
                FEATURE_CHECKPOINT_PATH.unlink()
        finally:
            cache.close()

    processed_set = set(processed_files)

    for index, file_path in enumerate(files):
        if file_path in processed_set:
            continue

        try:
            actor_id, emotion = parse_filename(file_path)
            feature_vector = extract_features(file_path)
            if feature_vector.shape != (EXPECTED_FEATURE_COUNT,):
                raise ValueError(f"Expected feature shape ({EXPECTED_FEATURE_COUNT},), got {feature_vector.shape}")

            features.append(feature_vector)
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
    print(f"Unique actors: {len(np.unique(actor_ids))}")

    if issue_records:
        print(f"\nWARNING: {len(issue_records)} files were skipped or had errors.")
        _write_issue_log(issue_records)

    missing_training_files = [file_path for file_path in files if file_path not in processed_set]
    if missing_training_files:
        for file_path in missing_training_files:
            _record_issue(issue_records, file_path, "missed", "not_processed")
        _save_feature_checkpoint(features, labels, actor_ids, processed_files, len(files), completed=False, issue_records=issue_records)
        raise RuntimeError(f"Feature extraction did not complete for {len(missing_training_files)} training files. See {SKIPPED_FILES_PATH}.")

    if len(X) != len(files):
        raise RuntimeError("Number of extracted samples does not match the number of training files.")

    print("\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
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
        feature_count=np.int64(EXPECTED_FEATURE_COUNT),
        feature_version=FEATURE_VERSION,
    )

    print(f"\nFinal feature cache saved: {FEATURE_CACHE_PATH}")
    if FEATURE_CHECKPOINT_PATH.exists():
        FEATURE_CHECKPOINT_PATH.unlink()

    _write_issue_log(issue_records)
    return X, y, actor_ids

# ============================================================
# ACTOR-AWARE DATA SPLIT
# ============================================================

def split_by_actor(X, y, actor_ids, train_size=0.70, validation_size=0.15, test_size=0.15):
    """Split the dataset by actor.

    Every recording belonging to the same actor is kept entirely inside one split.
    This prevents the same speaker/actor from appearing in both training and evaluation data."""

    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("train_size + validation_size + test_size must equal 1.0.")

    unique_actors = np.unique(actor_ids)

    if len(unique_actors) < 3:
        raise ValueError("At least 3 unique actors are required for an actor-aware split.")

    # --------------------------------------------------------
    # First split:
    #
    # 70% actors -> training
    # 30% actors -> temporary set
    # --------------------------------------------------------
    train_actors, temp_actors = train_test_split(unique_actors, train_size=train_size, random_state=RANDOM_STATE)

    # --------------------------------------------------------
    # Second split:
    #
    # Split the remaining 30% actors into:
    #
    # 15% validation
    # 15% test
    # --------------------------------------------------------
    test_relative_size = test_size / (validation_size + test_size)
    validation_actors, test_actors = train_test_split(temp_actors, test_size=test_relative_size, random_state=RANDOM_STATE)

    # --------------------------------------------------------
    # Create sample masks
    # --------------------------------------------------------
    train_mask = np.isin(actor_ids, train_actors)
    validation_mask = np.isin(actor_ids, validation_actors)
    test_mask = np.isin(actor_ids, test_actors)

    # --------------------------------------------------------
    # Explicit actor leakage checks
    # --------------------------------------------------------
    train_actor_set = set(train_actors)
    validation_actor_set = set(validation_actors)
    test_actor_set = set(test_actors)

    if train_actor_set & validation_actor_set:
        raise RuntimeError("Actor leakage detected between train and validation.")

    if train_actor_set & test_actor_set:
        raise RuntimeError("Actor leakage detected between train and test.")

    if validation_actor_set & test_actor_set:
        raise RuntimeError("Actor leakage detected between validation and test.")

    # --------------------------------------------------------
    # Build splits
    # --------------------------------------------------------

    X_train = X[train_mask]
    y_train = y[train_mask]
    train_actor_ids = actor_ids[train_mask]

    X_val = X[validation_mask]
    y_val = y[validation_mask]
    validation_actor_ids = actor_ids[validation_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]
    test_actor_ids = actor_ids[test_mask]

    # --------------------------------------------------------
    # Print split information
    # --------------------------------------------------------

    print("\nActor-aware split:")
    print(f"Training actors:   {len(train_actors)}")
    print(f"Validation actors: {len(validation_actors)}")
    print(f"Testing actors:    {len(test_actors)}")
    print("\nSample split:")
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples:    {len(X_test)}")
    print("\nActor leakage check: PASSED")

    return X_train, X_val, X_test, y_train, y_val, y_test, train_actor_ids, validation_actor_ids, test_actor_ids


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(input_features, num_classes):

    model = Sequential([
            Input(shape=(input_features,)),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax")
        ], name="speech_emotion_model")

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    X, y, actor_ids = load_dataset(DATASET_DIR)

    print("\nFinal feature shape:")
    print(f"X shape: {X.shape}")
    # --------------------------------------------------------
    # Actor-aware Train / Validation / Test split
    # --------------------------------------------------------

    X_train, X_val, X_test, y_train, y_val, y_test, train_actor_ids, validation_actor_ids, test_actor_ids = split_by_actor(X, y, actor_ids, train_size=TRAIN_SIZE, validation_size=VALIDATION_SIZE, test_size=TEST_SIZE)
    
    # The scaler is fitted only on training data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
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

    y_train_one_hot = to_categorical(y_train_encoded, num_classes=num_classes)
    y_val_one_hot = to_categorical(y_val_encoded, num_classes=num_classes)
    y_test_one_hot = to_categorical(y_test_encoded, num_classes=num_classes)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model = build_model(input_features=X_train.shape[1], num_classes=num_classes)
    model.summary()
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_log_dir = TENSORBOARD_LOG_DIR / run_name
    tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor="val_accuracy", mode="max", patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=7, min_lr=1e-6)
    history = model.fit(
        X_train,
        y_train_one_hot,
        validation_data=(X_val, y_val_one_hot),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=[tensorboard_callback, reduce_lr, early_stopping])
    
    loss, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("==============================")

    probabilities = model.predict(X_test, verbose=0)

    predictions = np.argmax(probabilities, axis=1)

    print("\nClassification Report:\n")
    print(classification_report(y_test_encoded, predictions, target_names=label_encoder.classes_))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test_encoded, predictions))

    model.save(MODEL_PATH)

    with open(SCALER_PATH, "wb") as file:
        pickle.dump(scaler, file)

    with open(ENCODER_PATH, "wb") as file:
        pickle.dump(label_encoder, file)

    feature_config = {
        "feature_count": EXPECTED_FEATURE_COUNT,
        "feature_version": FEATURE_VERSION,
        "features": {
            "mfcc": 40,
            "mfcc_delta": 40,
            "mfcc_delta_delta": 40,
            "chroma": 12,
            "mel": 128,
            "contrast": 7,
            "tonnetz": 6,
            "poly": 2
        },
        "emotions": label_encoder.classes_.tolist(),
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "split_strategy": "actor_aware",
        "random_state": RANDOM_STATE
    }

    with open(CONFIG_PATH, "w") as file:
        json.dump(feature_config, file, indent=4)

    # --------------------------------------------------------
    # Save training metadata
    # --------------------------------------------------------

    training_metadata = {
        "split_strategy": "actor_aware",
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "train_actor_ids": sorted(set(train_actor_ids.tolist())),
        "validation_actor_ids": sorted(set(validation_actor_ids.tolist())),
        "test_actor_ids": sorted(set(test_actor_ids.tolist()))
    }

    with open(METADATA_PATH, "w") as file:
        json.dump(training_metadata, file, indent=4)

    print(f"\nTensorBoard logs: {tensorboard_log_dir}")

    print("\nModel artifacts saved:")
    print(f"  {MODEL_PATH}")
    print(f"  {SCALER_PATH}")
    print(f"  {ENCODER_PATH}")
    print(f"  {CONFIG_PATH}")
    print(f"  {METADATA_PATH}")

if __name__ == "__main__":
    main()
