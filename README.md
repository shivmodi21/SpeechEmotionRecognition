# Speech Emotion Recognition Web App

A FastAPI + TensorFlow web application for **Speech Emotion Recognition
(SER)**. The application lets users upload an audio file, extracts the
same acoustic feature representation used during training, and predicts
one of **six emotions** using a trained neural network.

The current implementation is based on the **CREMA-D** audio dataset and
uses a 550-dimensional feature representation.

## What is implemented

- Upload an audio file from the browser using:
  - WAV
  - MP3
  - FLAC
  - OGG
  - M4A
  - AAC
- Drag-and-drop audio upload.
- Maximum upload size of **25 MB**.
- Send the audio file to the FastAPI `/api/predict` endpoint.
- Load the audio as mono at its native sample rate.
- Extract the current **550-dimensional acoustic feature vector**.
- Standardize the extracted feature vector using the training
`StandardScaler`.
- Load the trained Keras neural network and `LabelEncoder`.
- Return:
  - predicted emotion
  - confidence
  - probability for every emotion
  - uploaded filename
- While inference is running, the UI continuously cycles through all
six emotion chips. When prediction finishes, the animation stops
with the predicted emotion highlighted.
- Display the probability of every emotion using horizontal
probability bars.
- Explain the model pipeline, feature representation, and
neural-network architecture directly on the website.
- Cache extracted training features so subsequent training runs do not
need to recompute the complete dataset when the dataset and feature
configuration have not changed.
- Save feature-extraction checkpoints so a long extraction process can
be resumed after interruption.



## Supported emotions

The current model predicts six emotions:

1. **Angry**
2. **Disgust**
3. **Fear**
4. **Happy**
5. **Neutral**
6. **Sad**

The training code maps the CREMA-D filename emotion codes as follows:

  CREMA-D code   Emotion

---

  `ANG`          Angry
  `DIS`          Disgust
  `FEA`          Fear
  `NEU`          Neutral
  `SAD`          Sad
  `HAP`          Happy

## Feature extraction

The current feature extractor is **version** `v2` and produces exactly
**550 features**.

Unlike the earlier 195-feature representation, the current
implementation includes temporal information through **mean and standard
deviation statistics** and also includes MFCC delta and delta-delta
features.

### Feature composition

---

  Feature family        Components                                Features

---

  MFCC                  40 static MFCC + 40                            240
                        delta + 40  
                        delta-delta, each  
                        with mean and  
                        standard deviation    

  Chroma                12 chroma                                       24
                        coefficients with  
                        mean and standard  
                        deviation             

  Mel spectrogram       128 mel bands with                             256
                        mean and standard  
                        deviation             

  Spectral contrast     7 contrast                                      14
                        coefficients with  
                        mean and standard  
                        deviation             

  Tonnetz               6 Tonnetz features                              12
                        with mean and  
                        standard deviation    

  Polynomial features   2 polynomial spectral                            4
                        features with mean  
                        and standard  
                        deviation             

##   **Total**                                                        **550**

The feature extractor loads the audio as a mono waveform at its native
sample rate. It then computes the feature groups and summarizes each
time-varying feature using its mean and standard deviation.

### MFCC temporal features

The MFCC portion is now:

```text
40 static MFCC
+ 40 MFCC delta
+ 40 MFCC delta-delta
```

Each of these three groups contributes both mean and standard deviation:

```text
40 × 3 × 2 = 240 features
```

For unusually short audio recordings, the delta calculation
automatically reduces the window size when possible and safely falls
back to zero-valued deltas when the recording is too short.

## Model architecture

The current neural network is a dense feed-forward network:

```text
550 → 256 → 128 → 64 → 6
```

Architecture:

- Input: **550 features**
- Dense: **256 units**, ReLU
- Dropout: **0.30**
- Dense: **128 units**, ReLU
- Dropout: **0.30**
- Dense: **64 units**, ReLU
- Output: **6 units**, Softmax

The model is compiled with:

- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Metric:** Accuracy

Training configuration:

- Maximum epochs: **150**
- Batch size: **10**
- Test split: **15%**
- Validation split: **15%**
- Random state: **9**
- Early stopping based on validation accuracy
- Best weights restored after early stopping
- Learning rate reduction when validation accuracy stops improving
- TensorBoard logging enabled

The training pipeline uses stratified train/validation/test splits.

## Project structure

```text
SpeechEmotionRecognition/
├── backend/
│   ├── __init__.py
│   ├── feature_extraction.py
│   ├── main.py
│   ├── predictor.py
│   └── train_model.py
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── model/
│   ├── speech_emotion_model.keras   # generated by train_model.py
│   ├── scaler.pkl                   # generated by train_model.py
│   ├── label_encoder.pkl            # generated by train_model.py
│   └── feature_config.json          # generated by train_model.py
│
├── data/
│   ├── AudioWAV/                    # CREMA-D WAV files
│   └── features/
│       ├── crema_d_features.npz
│       └── extraction_checkpoint.npz
│
├── logs/                            # TensorBoard training logs
├── failed_files.json                # generated if feature extraction fails
├── requirements.txt
└── README.md
```



## 1. Create the Python environment

Python **3.10 or 3.11** is recommended for a straightforward TensorFlow
setup.

Create a virtual environment:

```bash
python -m venv .venv
```



### Windows

```bash
.venv\Scripts\activate
```



### Linux/macOS

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```



## 2. Prepare the dataset

The current training implementation expects the dataset at:

```text
data/AudioWAV/
```

The training code recursively searches this directory for `.wav` files.

For the current implementation, place the CREMA-D `AudioWAV` files
there:

```text
data/
└── AudioWAV/
    ├── 1001_DFA_ANG_XX.wav
    ├── 1001_DFA_DIS_XX.wav
    ├── 1001_DFA_FEA_XX.wav
    ├── ...
    └── ...
```

The emotion is extracted from the third underscore-separated component
of the filename. For example:

```text
1001_DFA_ANG_XX.wav
         ^^^
         ANG → angry
```

The training script validates the emotion code and will report an error
for an unknown code.

## 3. Train the model

From the project root, run:

```bash
python -m backend.train_model
```

The current training script does **not** take a `--data` command-line
argument. It uses the fixed dataset location:

```text
data/AudioWAV/
```



### Feature caching

Feature extraction can be time-consuming because the dataset contains
many audio files.

The training pipeline therefore stores a reusable feature cache at:

```text
data/features/crema_d_features.npz
```

The cache records:

- extracted features
- labels
- processed files
- total file count
- feature count
- feature version

The cache is reused only when it matches the current dataset and feature
configuration.

If the feature version or feature count changes, or the dataset files
change, the existing cache is considered incompatible and is rebuilt.

### Feature extraction checkpoints

During extraction, a checkpoint is periodically saved to:

```text
data/features/extraction_checkpoint.npz
```

The current implementation saves a checkpoint every **100 processed
files**.

If extraction is interrupted, a compatible checkpoint can be used to
resume extraction instead of starting from the beginning.

If any files fail during extraction, details are written to:

```text
failed_files.json
```

Training stops until the failed files are fixed and the process is run
again.

## 4. Generated model artifacts

After successful training, the following files are created in `model/`:

```text
model/
├── speech_emotion_model.keras
├── scaler.pkl
├── label_encoder.pkl
└── feature_config.json
```



### `speech_emotion_model.keras`

The trained TensorFlow/Keras neural network.

### `scaler.pkl`

The fitted `StandardScaler` used to standardize the training features.

The same scaler must be used during inference.

### `label_encoder.pkl`

The fitted scikit-learn `LabelEncoder` used to map between numeric model
outputs and emotion labels.

### `feature_config.json`

Stores the feature configuration used by the training run, including:

- feature count
- feature version
- feature-family counts
- emotion classes
- validation split
- test split
- random state



## 5. Start the website

From the project root:

```bash
uvicorn backend.main:app --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

The frontend is served directly by FastAPI, so a separate frontend
development server is not required.

## Prediction flow

The complete inference pipeline is:

```text
Audio File
    ↓
FastAPI Upload
    ↓
Temporary Audio File
    ↓
Librosa Audio Loading
    ↓
550-Dimensional Feature Extraction
    ↓
StandardScaler
    ↓
Trained Keras Model
    ↓
6-Class Softmax Probabilities
    ↓
LabelEncoder
    ↓
Predicted Emotion + Confidence + Probabilities
```

The API loads the model, scaler, and label encoder once and reuses them
for subsequent predictions.

## API



### `GET /api/health`

Returns a simple health response:

```json
{
  "status": "ok"
}
```



### `POST /api/predict`

Accepts an audio file as multipart form data:

```text
file=<audio file>
```

Supported extensions:

```text
.wav
.mp3
.flac
.ogg
.m4a
.aac
```

The server rejects files larger than **25 MB** and rejects empty files.

A successful response has the following structure:

```json
{
  "filename": "sample.wav",
  "emotion": "happy",
  "confidence": 0.91,
  "probabilities": {
    "angry": 0.02,
    "disgust": 0.01,
    "fear": 0.02,
    "happy": 0.91,
    "neutral": 0.03,
    "sad": 0.01
  }
}
```

The probability object contains all six model classes.

## Frontend prediction animation

While the `/api/predict` request is running, the frontend cycles
through:

```text
Angry
Disgust
Fear
Happy
Neutral
Sad
```

The highlighted chip changes every **320 ms**.

When the backend response arrives, the animation stops and the returned
emotion is highlighted. The result section then displays the predicted
emotion, confidence, filename, and all six probabilities.

## Model compatibility

The inference pipeline must use the same feature representation as the
model was trained on.

The current model expects:

```text
550 features
Feature version: v2
```

The model, scaler, label encoder, and feature configuration should
therefore be treated as one compatible set.

If you change the feature extraction logic, retrain the model and
regenerate the associated artifacts rather than mixing artifacts from
different training configurations.

In particular, do not use the older 195-feature artifacts with the
current 550-feature model.

## Technology stack



### Backend

- Python
- FastAPI
- Uvicorn
- TensorFlow / Keras
- Librosa
- NumPy
- scikit-learn



### Frontend

- HTML
- CSS
- JavaScript
- Font Awesome



## Notes

- The model predicts **six observed emotion classes**: Angry, Disgust,
Fear, Happy, Neutral, and Sad.
- The current feature representation contains **550 features**.
- The training dataset path is currently fixed to `data/AudioWAV/`.
- Training and inference must use the same feature-extraction
implementation.
- The uploaded audio is processed temporarily by the FastAPI backend
and the temporary file is removed after prediction.


---

## Author

**Shiv Modi** — B.Tech. + M.Tech., IIT Bombay  
[GitHub](https://github.com/shivmodi21) · [Portfolio](https://shivmodi21.github.io/) · [LinkedIn](https://www.linkedin.com/in/shivmodi210/)