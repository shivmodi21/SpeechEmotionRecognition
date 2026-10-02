import json
import numpy as np
import librosa


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

# Fixed number of time frames for every sample.
MAX_FRAMES = 256

# Expected CNN input shape:
# (mel_frequency_bins, time_frames, channels)
FEATURE_SHAPE = (N_MELS, MAX_FRAMES, 1)

# ============================================================
# Fixed-size spectrogram
# ============================================================

def _fix_time_dimension(mel, max_frames=MAX_FRAMES):
    """
    Make the spectrogram have exactly max_frames time frames.

    If the spectrogram is longer:
        Center crop.

    If it is shorter:
        Zero-pad on both sides.

    Input:
        mel -> shape (n_mels, time)

    Output:
        shape (n_mels, max_frames)
    """

    current_frames = mel.shape[1]

    if current_frames > max_frames:
        start = (current_frames - max_frames) // 2
        end = start + max_frames
        mel = mel[:, start:end]

    elif current_frames < max_frames:
        total_padding = max_frames - current_frames
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left

        mel = np.pad(mel, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)

    return mel


# ============================================================
# Log-Mel Spectrogram Extraction
# ============================================================

def extract_features(audio_path):
    """
    Extract a CNN-compatible log-Mel spectrogram.

    Pipeline:

        Audio
          ↓
        16 kHz resampling
          ↓
        Mel Spectrogram
          ↓
        Power → dB
          ↓
        Per-sample normalization
          ↓
        Fixed 128 * 256 representation
          ↓
        Add channel dimension

    Returns:
        numpy.ndarray

        Shape:
            (128, 256, 1)
    """

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    if y is None or len(y) == 0:
        raise ValueError(f"Audio file contains no samples: {audio_path}")

    y = y - np.mean(y)

    if not np.isfinite(y).all():
        raise ValueError(f"Audio contains NaN or Inf values: {audio_path}")

    if np.max(np.abs(y)) == 0:
        return np.zeros(FEATURE_SHAPE, dtype=np.float32)

    max_amplitude = np.max(np.abs(y))

    if max_amplitude > 0:
        y = y / max_amplitude

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=SAMPLE_RATE // 2,
        power=2.0
    )

    # --------------------------------------------------------
    # Convert power spectrogram to decibels
    # --------------------------------------------------------

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # --------------------------------------------------------
    # Normalize spectrogram
    #
    # This keeps the representation numerically stable
    # for CNN training.
    # --------------------------------------------------------

    mean = np.mean(mel_db)
    std = np.std(mel_db)

    mel_db = (mel_db - mean) / (std + 1e-8)

    # --------------------------------------------------------
    # Make time dimension fixed
    # --------------------------------------------------------

    mel_db = _fix_time_dimension(mel_db, MAX_FRAMES)

    # --------------------------------------------------------
    # Add CNN channel dimension
    #
    # (128, 256)
    #       ↓
    # (128, 256, 1)
    # --------------------------------------------------------

    features = np.expand_dims(mel_db, axis=-1)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if features.shape != FEATURE_SHAPE:
        raise ValueError(f"Unexpected feature shape for '{audio_path}'. Expected {FEATURE_SHAPE}, got {features.shape}")

    return features.astype(np.float32)


# ============================================================
# Metadata
# ============================================================

def get_feature_config():
    """
    Return feature extraction configuration.

    This is saved alongside the trained model so that
    prediction uses exactly the same preprocessing.
    """

    return {
        "feature_type": "log_mel_spectrogram",
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "max_frames": MAX_FRAMES,
        "feature_shape": list(FEATURE_SHAPE),
        "normalization": "per_sample_zscore",
        "mono": True,
        "fmax": SAMPLE_RATE // 2,
    }


# ============================================================
# Test extractor directly
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print("Usage:\npython -m backend.feature_extraction_cnn <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    print("=" * 60)
    print("CNN Feature Extraction Test")
    print("=" * 60)

    print(f"Audio file : {audio_path}")

    features = extract_features(audio_path)

    print(f"Feature shape : {features.shape}")
    print(f"Data type     : {features.dtype}")
    print(f"Min value     : {features.min():.4f}")
    print(f"Max value     : {features.max():.4f}")
    print(f"Mean          : {features.mean():.4f}")
    print(f"Std           : {features.std():.4f}")

    print("Feature configuration:")
    print(json.dumps(get_feature_config(), indent=4))

    print("Feature extraction successful.")