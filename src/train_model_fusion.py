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

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Concatenate,
)
from tensorflow.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ReduceLROnPlateau,
)

from .feature_extraction import extract_features as extract_handcrafted_features
from .feature_extraction_cnn import (
    extract_features as extract_cnn_features,
    FEATURE_SHAPE,
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_VERSION = "fusion_v1"
MODEL_NAME = "speech_emotion_fusion"

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "data" / "AudioWAV"
FEATURE_CACHE_DIR = BASE_DIR / "data" / "features"
MODEL_DIR = BASE_DIR / "model"
TENSORBOARD_LOG_DIR = BASE_DIR / "logs"

# ------------------------------------------------------------
# Fusion feature cache
# ------------------------------------------------------------

FEATURE_CACHE_PATH = (FEATURE_CACHE_DIR / f"features_{FEATURE_VERSION}.npz")
FEATURE_CHECKPOINT_PATH = (FEATURE_CACHE_DIR / f"feature_extraction_checkpoint_{FEATURE_VERSION}.npz")
SKIPPED_FILES_PATH = (BASE_DIR / "data" / f"skipped_files_{FEATURE_VERSION}.json")

# ------------------------------------------------------------
# Model artifacts
# ------------------------------------------------------------

MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}_{FEATURE_VERSION}.keras"
SCALER_PATH = MODEL_DIR / f"scaler_{FEATURE_VERSION}.pkl"
ENCODER_PATH = MODEL_DIR / f"label_encoder_{FEATURE_VERSION}.pkl"
CONFIG_PATH = MODEL_DIR / f"feature_config_{FEATURE_VERSION}.json"
METADATA_PATH = MODEL_DIR / f"training_metadata_{FEATURE_VERSION}.json"

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
TRAIN_SIZE = 1.0 - TEST_SIZE - VALIDATION_SIZE

RANDOM_STATE = 9

EPOCHS = 150
BATCH_SIZE = 32

CHECKPOINT_EVERY = 100

EXPECTED_HANDCRAFTED_FEATURE_COUNT = 550


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
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# FILENAME PARSING
# ============================================================

def parse_filename(file_path):
    """
    Parse a CREMA-D filename.

    Example:
        1001_DFA_ANG_XX.wav

    Returns:
        actor_id, emotion
    """

    filename = os.path.basename(file_path)
    filename = os.path.splitext(filename)[0]

    parts = filename.split("_")

    if len(parts) < 4:
        raise ValueError(
            f"Unexpected filename format: {filename}"
        )

    actor_id = parts[0]
    emotion_code = parts[2]

    if emotion_code not in EMOTION_MAP:
        return None, None

    return actor_id, EMOTION_MAP[emotion_code]


# ============================================================
# FEATURE CHECKPOINT
# ============================================================

def _save_feature_checkpoint(
    handcrafted_features,
    cnn_features,
    labels,
    actor_ids,
    processed_files,
    total_files,
    completed=False,
):
    """
    Save both feature representations together.

    Keeping the two representations in the same checkpoint
    guarantees that their sample ordering remains identical.
    """

    FEATURE_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        FEATURE_CHECKPOINT_PATH,

        handcrafted_X=np.asarray(
            handcrafted_features,
            dtype=np.float32,
        ),

        cnn_X=np.asarray(
            cnn_features,
            dtype=np.float32,
        ),

        y=np.asarray(labels),

        actor_ids=np.asarray(actor_ids),

        processed_files=np.asarray(
            processed_files
        ),

        total_files=np.int64(total_files),

        completed=np.bool_(completed),
    )


# ============================================================
# FEATURE CACHE VALIDATION
# ============================================================

def _is_valid_cache(cache, files):
    """
    Validate the completed fusion feature cache.
    """

    required_keys = {
        "handcrafted_X",
        "cnn_X",
        "y",
        "actor_ids",
        "processed_files",
        "total_files",
        "completed",
    }

    if not required_keys.issubset(
        set(cache.files)
    ):
        return False

    if not bool(cache["completed"]):
        return False

    cached_files = cache[
        "processed_files"
    ].tolist()

    if cached_files != files:
        return False

    if int(cache["total_files"]) != len(files):
        return False

    handcrafted_X = cache[
        "handcrafted_X"
    ]

    cnn_X = cache[
        "cnn_X"
    ]

    y = cache["y"]
    actor_ids = cache["actor_ids"]

    # --------------------------------------------------------
    # Handcrafted features
    # --------------------------------------------------------

    if handcrafted_X.ndim != 2:
        return False

    if handcrafted_X.shape[1] != (
        EXPECTED_HANDCRAFTED_FEATURE_COUNT
    ):
        return False

    # --------------------------------------------------------
    # CNN features
    # --------------------------------------------------------

    if cnn_X.ndim != 4:
        return False

    if tuple(cnn_X.shape[1:]) != tuple(
        FEATURE_SHAPE
    ):
        return False

    # --------------------------------------------------------
    # Sample counts
    # --------------------------------------------------------

    if len(handcrafted_X) != len(files):
        return False

    if len(cnn_X) != len(files):
        return False

    if len(y) != len(files):
        return False

    if len(actor_ids) != len(files):
        return False

    return True


# ============================================================
# NPZ LOADER
# ============================================================

def _load_npz(path):
    """
    Load an NPZ file and close it immediately.
    """

    cache = np.load(
        path,
        allow_pickle=False,
    )

    try:
        data = {
            key: cache[key]
            for key in cache.files
        }
    finally:
        cache.close()

    return data


# ============================================================
# DATASET / FEATURE EXTRACTION
# ============================================================

def load_dataset(dataset_dir):
    """
    Extract both feature representations.

    Returns:

        handcrafted_X
            Shape: (samples, 550)

        cnn_X
            Shape: (samples, 128, 256, 1)

        y
            Emotion labels

        actor_ids
            Actor IDs
    """

    pattern = os.path.join(
        dataset_dir,
        "**",
        "*.wav",
    )

    files = sorted(
        glob.glob(
            pattern,
            recursive=True,
        )
    )

    # --------------------------------------------------------
    # Keep only the four selected emotions
    # --------------------------------------------------------

    selected_files = []

    for file_path in files:

        actor_id, emotion = parse_filename(
            file_path
        )

        if emotion is None:
            continue

        selected_files.append(
            file_path
        )

    files = selected_files

    print(
        f"Found {len(files)} selected WAV files."
    )

    if not files:
        raise FileNotFoundError(
            f"No WAV files found in: {dataset_dir}"
        )

    # ========================================================
    # COMPLETED CACHE
    # ========================================================

    if FEATURE_CACHE_PATH.exists():

        cache = _load_npz(
            FEATURE_CACHE_PATH
        )

        if _is_valid_cache(
            cache,
            files,
        ):

            handcrafted_X = cache[
                "handcrafted_X"
            ].astype(
                np.float32,
                copy=False,
            )

            cnn_X = cache[
                "cnn_X"
            ].astype(
                np.float32,
                copy=False,
            )

            y = cache["y"]
            actor_ids = cache[
                "actor_ids"
            ]

            print(
                f"\nLoaded cached fusion features:"
                f"\n{FEATURE_CACHE_PATH}"
            )

            print(
                f"Handcrafted shape: "
                f"{handcrafted_X.shape}"
            )

            print(
                f"CNN shape: "
                f"{cnn_X.shape}"
            )

            print(
                f"Unique actors: "
                f"{len(np.unique(actor_ids))}"
            )

            return (
                handcrafted_X,
                cnn_X,
                y,
                actor_ids,
            )

        print(
            "\nExisting fusion feature cache "
            "is incompatible."
        )

        print(
            "Rebuilding fusion feature cache..."
        )

        FEATURE_CACHE_PATH.unlink()

    # ========================================================
    # RESUME FROM CHECKPOINT
    # ========================================================

    handcrafted_features = []
    cnn_features = []
    labels = []
    actor_ids = []
    processed_files = []

    start_index = 0

    if FEATURE_CHECKPOINT_PATH.exists():

        cache = _load_npz(
            FEATURE_CHECKPOINT_PATH
        )

        required_keys = {
            "handcrafted_X",
            "cnn_X",
            "y",
            "actor_ids",
            "processed_files",
            "total_files",
        }

        metadata_valid = (
            required_keys.issubset(
                set(cache.keys())
            )
        )

        cached_files = cache[
            "processed_files"
        ].tolist()

        cached_total = int(
            cache["total_files"]
        )

        dataset_valid = (
            cached_total == len(files)
            and files[
                :len(cached_files)
            ] == cached_files
        )

        shape_valid = (
            cache[
                "handcrafted_X"
            ].ndim == 2
            and cache[
                "handcrafted_X"
            ].shape[1]
            == EXPECTED_HANDCRAFTED_FEATURE_COUNT
            and cache[
                "cnn_X"
            ].ndim == 4
            and tuple(
                cache[
                    "cnn_X"
                ].shape[1:]
            ) == tuple(FEATURE_SHAPE)
            and len(
                cache[
                    "handcrafted_X"
                ]
            )
            == len(
                cache[
                    "cnn_X"
                ]
            )
            and len(
                cache[
                    "handcrafted_X"
                ]
            )
            == len(
                cache["y"]
            )
            and len(
                cache[
                    "handcrafted_X"
                ]
            )
            == len(
                cache["actor_ids"]
            )
            and len(
                cache[
                    "handcrafted_X"
                ]
            )
            == len(
                cached_files
            )
        )

        if (
            metadata_valid
            and dataset_valid
            and shape_valid
        ):

            handcrafted_features = list(
                cache[
                    "handcrafted_X"
                ]
            )

            cnn_features = list(
                cache["cnn_X"]
            )

            labels = list(
                cache["y"]
            )

            actor_ids = list(
                cache["actor_ids"]
            )

            processed_files = cached_files

            start_index = len(
                processed_files
            )

            print(
                f"\nResuming feature extraction "
                f"from {start_index}/{len(files)} files."
            )

        else:

            print(
                "\nExisting fusion checkpoint "
                "is incompatible."
            )

            print(
                "Starting feature extraction "
                "from scratch."
            )

            FEATURE_CHECKPOINT_PATH.unlink()

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    failed_files = []

    for index in range(
        start_index,
        len(files),
    ):

        file_path = files[index]

        try:

            actor_id, emotion = (
                parse_filename(
                    file_path
                )
            )

            # ------------------------------------------------
            # Handcrafted features
            # ------------------------------------------------

            handcrafted_vector = (
                extract_handcrafted_features(
                    file_path
                )
            )

            if handcrafted_vector.shape != (
                EXPECTED_HANDCRAFTED_FEATURE_COUNT,
            ):
                raise ValueError(
                    "Unexpected handcrafted "
                    f"feature shape: "
                    f"{handcrafted_vector.shape}"
                )

            # ------------------------------------------------
            # CNN features
            # ------------------------------------------------

            cnn_tensor = (
                extract_cnn_features(
                    file_path
                )
            )

            if cnn_tensor.shape != (
                FEATURE_SHAPE
            ):
                raise ValueError(
                    "Unexpected CNN feature "
                    f"shape: "
                    f"{cnn_tensor.shape}"
                )

            # ------------------------------------------------
            # Zero-feature CNN sample
            # ------------------------------------------------

            if not np.any(cnn_tensor):

                print(
                    f"Skipping zero-feature file: "
                    f"{file_path}"
                )

                processed_files.append(
                    file_path
                )

                continue

            # ------------------------------------------------
            # Store both representations
            # ------------------------------------------------

            handcrafted_features.append(
                handcrafted_vector
            )

            cnn_features.append(
                cnn_tensor
            )

            labels.append(
                emotion
            )

            actor_ids.append(
                actor_id
            )

            processed_files.append(
                file_path
            )

        except Exception as error:

            print(
                f"\nFailed: {file_path}"
            )

            print(
                f"Error Type: "
                f"{type(error).__name__}"
            )

            print(
                f"Reason: {repr(error)}"
            )

            failed_files.append(
                {
                    "file": file_path,
                    "error_type": (
                        type(error).__name__
                    ),
                    "error": repr(error),
                }
            )

        current_count = index + 1

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            current_count
            % CHECKPOINT_EVERY
            == 0
        ):

            _save_feature_checkpoint(
                handcrafted_features,
                cnn_features,
                labels,
                actor_ids,
                processed_files,
                len(files),
                completed=False,
            )

            print(
                f"Processed "
                f"{current_count}/{len(files)} "
                f"files - checkpoint saved."
            )

    # ========================================================
    # CONVERT TO ARRAYS
    # ========================================================

    handcrafted_X = np.asarray(
        handcrafted_features,
        dtype=np.float32,
    )

    cnn_X = np.asarray(
        cnn_features,
        dtype=np.float32,
    )

    y = np.asarray(
        labels
    )

    actor_ids = np.asarray(
        actor_ids
    )

    print(
        "\nDataset loaded."
    )

    print(
        f"Handcrafted X shape: "
        f"{handcrafted_X.shape}"
    )

    print(
        f"CNN X shape: "
        f"{cnn_X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    print(
        f"Actor IDs shape: "
        f"{actor_ids.shape}"
    )

    # ========================================================
    # FAILED FILES
    # ========================================================

    if failed_files:

        print(
            f"\nWARNING: "
            f"{len(failed_files)} files failed."
        )

        with open(
            SKIPPED_FILES_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                failed_files,
                file,
                indent=4,
            )

        _save_feature_checkpoint(
            handcrafted_features,
            cnn_features,
            labels,
            actor_ids,
            processed_files,
            len(files),
            completed=False,
        )

        raise RuntimeError(
            "Feature extraction failed for "
            f"{len(failed_files)} files. "
            "Fix the failed files and rerun."
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    # The number of extracted samples can be
    # smaller than the original selected files
    # because zero-feature files are intentionally
    # skipped.

    if len(handcrafted_X) != len(cnn_X):
        raise RuntimeError(
            "Handcrafted and CNN feature counts "
            "do not match."
        )

    if len(handcrafted_X) != len(y):
        raise RuntimeError(
            "Feature and label counts do not match."
        )

    if len(handcrafted_X) != len(actor_ids):
        raise RuntimeError(
            "Feature and actor ID counts "
            "do not match."
        )

    if handcrafted_X.shape[1] != (
        EXPECTED_HANDCRAFTED_FEATURE_COUNT
    ):
        raise RuntimeError(
            "Unexpected handcrafted feature count."
        )

    if cnn_X.shape[1:] != FEATURE_SHAPE:
        raise RuntimeError(
            "Unexpected CNN feature shape."
        )

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print(
        "\nClass distribution:"
    )

    unique, counts = np.unique(
        y,
        return_counts=True,
    )

    for emotion, count in zip(
        unique,
        counts,
    ):

        print(
            f"{emotion:10s}: {count}"
        )

    # ========================================================
    # SAVE COMPLETED CACHE
    # ========================================================

    _save_feature_checkpoint(
        handcrafted_features,
        cnn_features,
        labels,
        actor_ids,
        processed_files,
        len(files),
        completed=True,
    )

    FEATURE_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        FEATURE_CACHE_PATH,

        handcrafted_X=handcrafted_X,

        cnn_X=cnn_X,

        y=y,

        actor_ids=actor_ids,

        processed_files=np.asarray(
            processed_files
        ),

        total_files=np.int64(
            len(files)
        ),

        completed=np.bool_(True),
    )

    print(
        f"\nFusion feature cache saved:"
        f"\n{FEATURE_CACHE_PATH}"
    )

    if (
        FEATURE_CHECKPOINT_PATH.exists()
    ):

        FEATURE_CHECKPOINT_PATH.unlink()

    return (
        handcrafted_X,
        cnn_X,
        y,
        actor_ids,
    )


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
    Split actors first and then create sample indices.

    No actor can appear in more than one split.
    """

    unique_actors = np.unique(
        actor_ids
    )

    if len(unique_actors) < 3:
        raise ValueError(
            "At least 3 unique actors "
            "are required."
        )

    # --------------------------------------------------------
    # Train + validation / test
    # --------------------------------------------------------

    train_val_actors, test_actors = (
        train_test_split(
            unique_actors,
            test_size=test_size,
            random_state=random_state,
        )
    )

    # --------------------------------------------------------
    # Train / validation
    # --------------------------------------------------------

    validation_relative_size = (
        validation_size
        / (1.0 - test_size)
    )

    train_actors, validation_actors = (
        train_test_split(
            train_val_actors,
            test_size=validation_relative_size,
            random_state=random_state,
        )
    )

    train_actors = set(
        train_actors.tolist()
    )

    validation_actors = set(
        validation_actors.tolist()
    )

    test_actors = set(
        test_actors.tolist()
    )

    # --------------------------------------------------------
    # Sample indices
    # --------------------------------------------------------

    train_indices = np.array(
        [
            index
            for index, actor
            in enumerate(actor_ids)
            if actor in train_actors
        ],
        dtype=np.int64,
    )

    validation_indices = np.array(
        [
            index
            for index, actor
            in enumerate(actor_ids)
            if actor in validation_actors
        ],
        dtype=np.int64,
    )

    test_indices = np.array(
        [
            index
            for index, actor
            in enumerate(actor_ids)
            if actor in test_actors
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Explicit leakage checks
    # --------------------------------------------------------

    if not train_actors.isdisjoint(
        validation_actors
    ):
        raise RuntimeError(
            "Actor leakage detected between "
            "train and validation."
        )

    if not train_actors.isdisjoint(
        test_actors
    ):
        raise RuntimeError(
            "Actor leakage detected between "
            "train and test."
        )

    if not validation_actors.isdisjoint(
        test_actors
    ):
        raise RuntimeError(
            "Actor leakage detected between "
            "validation and test."
        )

    return (
        train_indices,
        validation_indices,
        test_indices,
        sorted(train_actors),
        sorted(validation_actors),
        sorted(test_actors),
    )


# ============================================================
# BUILD FUSION MODEL
# ============================================================

def build_model(
    handcrafted_input_size,
    cnn_input_shape,
    num_classes,
):
    """
    Two-branch feature-fusion network.

    Branch 1:
        550 handcrafted features
        -> Dense 256
        -> Dense 128

    Branch 2:
        Log-Mel spectrogram
        -> CNN
        -> Global Average Pooling
        -> Dense 128

    Fusion:
        Concatenate
        -> Dense 128
        -> Dropout
        -> Dense 64
        -> Softmax
    """

    # ========================================================
    # HANDCRAFTED FEATURE BRANCH
    # ========================================================

    handcrafted_input = Input(
        shape=(handcrafted_input_size,),
        name="handcrafted_input",
    )

    handcrafted = Dense(
        256,
        activation="relu",
        name="handcrafted_dense_256",
    )(handcrafted_input)

    handcrafted = Dropout(
        0.3,
        name="handcrafted_dropout",
    )(handcrafted)

    handcrafted = Dense(
        128,
        activation="relu",
        name="handcrafted_dense_128",
    )(handcrafted)

    # ========================================================
    # CNN BRANCH
    # ========================================================

    cnn_input = Input(
        shape=cnn_input_shape,
        name="cnn_input",
    )

    cnn = Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="cnn_conv_32",
    )(cnn_input)

    cnn = BatchNormalization(
        name="cnn_bn_32",
    )(cnn)

    cnn = MaxPooling2D(
        pool_size=(2, 2),
        name="cnn_pool_32",
    )(cnn)

    cnn = Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="cnn_conv_64",
    )(cnn)

    cnn = BatchNormalization(
        name="cnn_bn_64",
    )(cnn)

    cnn = MaxPooling2D(
        pool_size=(2, 2),
        name="cnn_pool_64",
    )(cnn)

    cnn = Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="cnn_conv_128",
    )(cnn)

    cnn = BatchNormalization(
        name="cnn_bn_128",
    )(cnn)

    cnn = MaxPooling2D(
        pool_size=(2, 2),
        name="cnn_pool_128",
    )(cnn)

    cnn = GlobalAveragePooling2D(
        name="cnn_global_average_pooling",
    )(cnn)

    cnn = Dense(
        128,
        activation="relu",
        name="cnn_dense_128",
    )(cnn)

    # ========================================================
    # FEATURE FUSION
    # ========================================================

    fused = Concatenate(
        name="feature_fusion",
    )(
        [
            handcrafted,
            cnn,
        ]
    )

    fused = Dense(
        128,
        activation="relu",
        name="fusion_dense_128",
    )(fused)

    fused = Dropout(
        0.4,
        name="fusion_dropout",
    )(fused)

    fused = Dense(
        64,
        activation="relu",
        name="fusion_dense_64",
    )(fused)

    output = Dense(
        num_classes,
        activation="softmax",
        name="emotion_output",
    )(fused)

    model = Model(
        inputs=[
            handcrafted_input,
            cnn_input,
        ],
        outputs=output,
        name="speech_emotion_fusion_model",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD / EXTRACT FEATURES
    # ========================================================

    (
        handcrafted_X,
        cnn_X,
        y,
        actor_ids,
    ) = load_dataset(
        DATASET_DIR
    )

    print(
        "\n=============================="
    )

    print(
        "FINAL FEATURE SHAPES"
    )

    print(
        "=============================="
    )

    print(
        f"Handcrafted: "
        f"{handcrafted_X.shape}"
    )

    print(
        f"CNN: "
        f"{cnn_X.shape}"
    )

    # ========================================================
    # ACTOR-AWARE SPLIT
    # ========================================================

    (
        train_indices,
        validation_indices,
        test_indices,
        train_actors,
        validation_actors,
        test_actors,
    ) = create_actor_aware_split(
        actor_ids
    )

    # --------------------------------------------------------
    # Split handcrafted features
    # --------------------------------------------------------

    X_hand_train = handcrafted_X[
        train_indices
    ]

    X_hand_validation = handcrafted_X[
        validation_indices
    ]

    X_hand_test = handcrafted_X[
        test_indices
    ]

    # --------------------------------------------------------
    # Split CNN features
    # --------------------------------------------------------

    X_cnn_train = cnn_X[
        train_indices
    ]

    X_cnn_validation = cnn_X[
        validation_indices
    ]

    X_cnn_test = cnn_X[
        test_indices
    ]

    # --------------------------------------------------------
    # Split labels
    # --------------------------------------------------------

    y_train = y[
        train_indices
    ]

    y_validation = y[
        validation_indices
    ]

    y_test = y[
        test_indices
    ]

    print(
        "\n=============================="
    )

    print(
        "ACTOR-AWARE SPLIT"
    )

    print(
        "=============================="
    )

    print(
        f"Training actors:   "
        f"{len(train_actors)}"
    )

    print(
        f"Validation actors: "
        f"{len(validation_actors)}"
    )

    print(
        f"Testing actors:    "
        f"{len(test_actors)}"
    )

    print(
        f"Training samples:   "
        f"{len(train_indices)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_indices)}"
    )

    print(
        f"Testing samples:    "
        f"{len(test_indices)}"
    )

    print(
        "\nActor overlap check:"
    )

    print(
        "Train ∩ Validation: "
        f"{len(set(train_actors) & set(validation_actors))}"
    )

    print(
        "Train ∩ Test: "
        f"{len(set(train_actors) & set(test_actors))}"
    )

    print(
        "Validation ∩ Test: "
        f"{len(set(validation_actors) & set(test_actors))}"
    )

    # ========================================================
    # SCALE HANDCRAFTED FEATURES
    # ========================================================

    # IMPORTANT:
    # Fit only on the training set.

    scaler = StandardScaler()

    X_hand_train = scaler.fit_transform(
        X_hand_train
    )

    X_hand_validation = scaler.transform(
        X_hand_validation
    )

    X_hand_test = scaler.transform(
        X_hand_test
    )

    # CNN features are already normalized
    # by feature_extraction_cnn.py and are
    # therefore passed directly to the model.

    # ========================================================
    # ENCODE LABELS
    # ========================================================

    label_encoder = LabelEncoder()

    y_train_encoded = (
        label_encoder.fit_transform(
            y_train
        )
    )

    y_validation_encoded = (
        label_encoder.transform(
            y_validation
        )
    )

    y_test_encoded = (
        label_encoder.transform(
            y_test
        )
    )

    num_classes = len(
        label_encoder.classes_
    )

    print(
        "\nClasses:"
    )

    for index, emotion in enumerate(
        label_encoder.classes_
    ):

        print(
            f"{index}: {emotion}"
        )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = build_model(
        handcrafted_input_size=(
            EXPECTED_HANDCRAFTED_FEATURE_COUNT
        ),
        cnn_input_shape=FEATURE_SHAPE,
        num_classes=num_classes,
    )

    model.summary()

    # ========================================================
    # CALLBACKS
    # ========================================================

    run_name = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    tensorboard_log_dir = (
        TENSORBOARD_LOG_DIR
        / f"fusion_{run_name}"
    )

    tensorboard_callback = TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_accuracy",
        mode="max",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1,
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "TRAINING FUSION MODEL"
    )

    print(
        "=============================="
    )

    history = model.fit(
        {
            "handcrafted_input": X_hand_train,
            "cnn_input": X_cnn_train,
        },
        y_train_encoded,

        validation_data=(
            {
                "handcrafted_input":
                    X_hand_validation,

                "cnn_input":
                    X_cnn_validation,
            },
            y_validation_encoded,
        ),

        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,

        callbacks=[
            tensorboard_callback,
            reduce_lr,
            early_stopping,
        ],
    )

    # ========================================================
    # TEST
    # ========================================================

    test_loss, test_accuracy = (
        model.evaluate(
            {
                "handcrafted_input":
                    X_hand_test,

                "cnn_input":
                    X_cnn_test,
            },
            y_test_encoded,
            verbose=0,
        )
    )

    print(
        "\n=============================="
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=============================="
    )

    print(
        f"Test Loss:     "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    probabilities = model.predict(
        {
            "handcrafted_input":
                X_hand_test,

            "cnn_input":
                X_cnn_test,
        },
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            y_test_encoded,
            predictions,
            target_names=(
                label_encoder.classes_
            ),
        )
    )

    print(
        "\nConfusion Matrix:\n"
    )

    print(
        confusion_matrix(
            y_test_encoded,
            predictions,
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model.save(
        MODEL_PATH
    )

    with open(
        SCALER_PATH,
        "wb",
    ) as file:

        pickle.dump(
            scaler,
            file,
        )

    with open(
        ENCODER_PATH,
        "wb",
    ) as file:

        pickle.dump(
            label_encoder,
            file,
        )

    # ========================================================
    # FEATURE CONFIG
    # ========================================================

    feature_config = {
        "model_type":
            "two_branch_feature_fusion",

        "dataset":
            "CREMA-D",

        "emotions":
            label_encoder.classes_.tolist(),

        "handcrafted_features": {
            "feature_count":
                EXPECTED_HANDCRAFTED_FEATURE_COUNT,

            "feature_type":
                "handcrafted_audio_features",

            "mfcc":
                40,

            "mfcc_delta":
                40,

            "mfcc_delta_delta":
                40,

            "chroma":
                12,

            "mel":
                128,

            "contrast":
                7,

            "tonnetz":
                6,

            "poly":
                2,
        },

        "cnn_features": {
            "feature_type":
                "log_mel_spectrogram",

            "feature_shape":
                list(FEATURE_SHAPE),
        },

        "fusion": {
            "handcrafted_branch_output":
                128,

            "cnn_branch_output":
                128,

            "fusion_input_size":
                256,

            "fusion_dense":
                128,
        },

        "validation_size":
            VALIDATION_SIZE,

        "test_size":
            TEST_SIZE,

        "train_size":
            TRAIN_SIZE,

        "random_state":
            RANDOM_STATE,

        "split_strategy":
            "actor_aware",

        "handcrafted_scaling":
            "StandardScaler_fit_on_training_only",
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            feature_config,
            file,
            indent=4,
        )

    # ========================================================
    # TRAINING METADATA
    # ========================================================

    training_metadata = {
        "model_type":
            "two_branch_feature_fusion",

        "dataset":
            "CREMA-D",

        "dataset_path":
            str(DATASET_DIR),

        "random_state":
            RANDOM_STATE,

        "split_strategy":
            "actor_aware",

        "train_size":
            TRAIN_SIZE,

        "validation_size":
            VALIDATION_SIZE,

        "test_size":
            TEST_SIZE,

        "train_samples":
            int(len(X_hand_train)),

        "validation_samples":
            int(len(X_hand_validation)),

        "test_samples":
            int(len(X_hand_test)),

        "train_actors":
            train_actors,

        "validation_actors":
            validation_actors,

        "test_actors":
            test_actors,

        "handcrafted_feature_count":
            EXPECTED_HANDCRAFTED_FEATURE_COUNT,

        "cnn_feature_shape":
            list(FEATURE_SHAPE),

        "epochs_requested":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "test_loss":
            float(test_loss),

        "test_accuracy":
            float(test_accuracy),

        "training_completed_at":
            datetime.now().isoformat(),

        "tensorboard_log_dir":
            str(tensorboard_log_dir),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            training_metadata,
            file,
            indent=4,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        f"\nTensorBoard logs:"
        f"\n{tensorboard_log_dir}"
    )

    print(
        "\nModel artifacts saved:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    print(
        f"  {SCALER_PATH}"
    )

    print(
        f"  {ENCODER_PATH}"
    )

    print(
        f"  {CONFIG_PATH}"
    )

    print(
        f"  {METADATA_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()