import numpy as np
import librosa

def _mean_and_std(features):
    """Return mean and standard deviation for each feature across time."""
    features = np.asarray(features, dtype=np.float32)

    mean = np.mean(features, axis=1)
    std = np.std(features, axis=1)

    return np.hstack((mean, std))


def _safe_delta(features, order=1):
    """
    Calculate temporal deltas while handling short audio sequences.

    Librosa's default delta width is 9. For unusually short recordings,
    reduce the width to the largest valid odd value.
    """
    n_frames = features.shape[1]

    if n_frames < 3:
        return np.zeros_like(features, dtype=np.float32)

    width = min(9, n_frames)

    if width % 2 == 0:
        width -= 1

    if width < 3:
        return np.zeros_like(features, dtype=np.float32)

    return librosa.feature.delta(features, order=order, width=width).astype(np.float32)


def extract_features(
    file_name,
    mfcc=True,
    chroma=True,
    mel=True,
    contrast=True,
    tonnetz=True,
    poly=True
):
    """
    Extract the updated speech-emotion feature representation.

    Feature groups
    --------------
    MFCC:
        40 static MFCC
        40 first-order MFCC delta
        40 second-order MFCC delta-delta

    For each feature group, mean and standard deviation over time
    are retained.

    Feature counts
    --------------
    MFCC:
        40 x 3 x 2 = 240

    Chroma:
        12 x 2 = 24

    Mel:
        128 x 2 = 256

    Spectral contrast:
        7 x 2 = 14

    Tonnetz:
        6 x 2 = 12

    Polynomial features:
        2 x 2 = 4

    Total:
        550 features
    """
    X, sample_rate = librosa.load(file_name, sr=None, mono=True)

    if X.size == 0:
        raise ValueError("The audio file contains no samples.")

    stft = None

    if chroma or contrast or poly:
        stft = np.abs(librosa.stft(X))

    feature_groups = []

    # --------------------------------------------------------
    # MFCC + Delta + Delta-Delta
    # --------------------------------------------------------
    if mfcc:
        mfcc_values = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        mfcc_delta = _safe_delta(mfcc_values, order=1)
        mfcc_delta_delta = _safe_delta(mfcc_values, order=2)
        feature_groups.extend([_mean_and_std(mfcc_values), _mean_and_std(mfcc_delta), _mean_and_std(mfcc_delta_delta)])

    if chroma:
        chroma_values = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        feature_groups.append(_mean_and_std(chroma_values))

    if mel:
        mel_values = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        feature_groups.append(_mean_and_std(mel_values))

    if contrast:
        contrast_values = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
        feature_groups.append(_mean_and_std(contrast_values))

    if tonnetz:
        harmonic = librosa.effects.harmonic(X)

        tonnetz_values = librosa.feature.tonnetz(y=harmonic, sr=sample_rate)
        feature_groups.append(_mean_and_std(tonnetz_values))

    if poly:
        poly_values = librosa.feature.poly_features(S=stft)

        feature_groups.append(_mean_and_std(poly_values))

    result = np.hstack(feature_groups).astype(np.float32)

    if result.shape[0] != 550:
        raise ValueError(f"Expected 550 features, but extracted {result.shape[0]}.")

    return result
