# Speech Emotion Recognition

A web-based **Speech Emotion Recognition (SER)** project that uses a deep-learning **fusion model** to recognize emotions from speech audio.

The model combines two complementary representations of an audio signal:

- **550 handcrafted acoustic features**
- **128 × 256 × 1 log-Mel spectrogram features processed by a CNN**

The two representations are fused before the final prediction. The current model recognizes four emotions:

**Angry · Happy · Neutral · Sad**

### Live Demo

[Speech Emotion Recognition](https://speechemotionrecognition-box6.onrender.com/)

---



## Skills & Techniques



### Machine Learning / Deep Learning

- Speech Emotion Recognition
- Feature engineering
- Convolutional Neural Networks (CNN)
- Multi-branch feature fusion
- Dense neural networks
- Softmax classification
- StandardScaler
- Label encoding
- Actor-aware train/validation/test splitting



### Audio Processing

- Python
- Librosa
- MFCC
- MFCC delta and delta-delta
- Chroma
- Mel spectrogram
- Spectral contrast
- Tonnetz
- Polynomial spectral features
- Log-Mel spectrograms



### Development

- TensorFlow / Keras
- NumPy
- scikit-learn
- FastAPI
- HTML
- CSS
- JavaScript
- Uvicorn
- Git / GitHub
- Render

---



## Features



### Audio Prediction

- Upload an audio file directly from the browser
- Drag-and-drop support
- Supports WAV, MP3, FLAC, OGG, M4A and AAC
- Maximum upload size of 25 MB
- Real-time prediction status



### Prediction Results

- Predicted emotion
- Prediction confidence
- Probability distribution across the four emotions
- Animated emotion indicators while prediction is in progress



### Model Information

The website also presents:

- Feature composition
- CNN representation
- Fusion architecture
- Prediction pipeline
- Model technology stack

---



## Feature Representation

The model uses two different representations of the same speech signal.

### Handcrafted Acoustic Features — 550


| Feature             | Components                                                      | Features |
| ------------------- | --------------------------------------------------------------- | -------- |
| MFCC                | 40 MFCC + delta + delta-delta, with mean and standard deviation | 240      |
| Chroma              | 12 features with mean and standard deviation                    | 24       |
| Mel Spectrogram     | 128 features with mean and standard deviation                   | 256      |
| Spectral Contrast   | 7 features with mean and standard deviation                     | 14       |
| Tonnetz             | 6 features with mean and standard deviation                     | 12       |
| Polynomial Features | 2 features with mean and standard deviation                     | 4        |
| **Total**           |                                                                 | **550**  |




### CNN Features

The CNN branch uses a fixed-size log-Mel spectrogram:

```text
128 × 256 × 1
```

Audio is converted into a normalized log-Mel representation and adjusted to a fixed time dimension so that every sample has the same input shape.

---



## Model Structure

The project uses a **two-branch feature-fusion neural network**.

```text
                    Audio
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
      550 Acoustic        Log-Mel Spectrogram
         Features             128×256×1
             │                 │
             ▼                 ▼
        Dense 256          Conv2D 32
        Dropout 0.3        BatchNorm
        Dense 128          MaxPooling
                              │
                           Conv2D 64
                           BatchNorm
                           MaxPooling
                              │
                          Conv2D 128
                           BatchNorm
                           MaxPooling
                              │
                      Global Average Pooling
                              │
                          Dense 128
             │                 │
             └───────┬─────────┘
                     ▼
                Concatenate
                     │
                  Dense 128
                 Dropout 0.4
                  Dense 64
                     │
                Softmax Output
                     │
              4 Emotion Classes
```



### Training Configuration

- Optimizer: **Adam**
- Loss: **Sparse Categorical Crossentropy**
- Metric: **Accuracy**
- Batch size: **32**
- Maximum epochs: **150**
- Early stopping
- Learning-rate reduction
- Actor-aware data splitting
- StandardScaler fitted only on the training data

---



## Prediction Pipeline

```text
Audio File
    ↓
Audio Upload
    ↓
FastAPI Backend
    ↓
┌───────────────────────────────┐
│       Feature Extraction      │
├────────────────┬──────────────┤
│                │              │
▼                ▼              │
550 Acoustic     Log-Mel        │
Features         Spectrogram    │
│                │              │
▼                ▼              │
StandardScaler   CNN            │
│                │              │
└────────┬───────┘              │
         ▼
   Feature Fusion
         ↓
   Neural Network
         ↓
    4-Class Softmax
         ↓
Predicted Emotion
         ↓
     Web Interface
```

---



## Project Structure

```text
SpeechEmotionRecognition/
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   └── predictor.py
│
├── data/
│   └── AudioWAV/
│
├── frontend/
│   ├── images/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── model/
│   ├── helper/
│   ├── metadata/
│   ├── speech_emotion_model_cnn_v1.keras
│   ├── speech_emotion_model_fusion_v1.keras
│   └── speech_emotion_model_v3.keras
│
├── src/
│   ├── feature_extraction.py
│   ├── feature_extraction_cnn.py
│   ├── train_model.py
│   ├── train_model_cnn.py
│   └── train_model_fusion.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---



## Run Locally



### 1. Clone the repository

```bash
git clone https://github.com/shivmodi21/SpeechEmotionRecognition
cd SpeechEmotionRecognition
```



### 2. Create a virtual environment



#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```



#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```



### 3. Install dependencies

```bash
pip install -r requirements.txt
```



### 4. Start the application

From the project root:

```bash
uvicorn backend.main:app --reload
```



### 5. Open the website

```text
http://127.0.0.1:8000
```

The FastAPI backend serves the frontend, so a separate frontend development server is not required.

---



## Dataset

The project uses the **CREMA-D** speech emotion dataset.

For training, the dataset is expected at:

```text
data/AudioWAV/
```

The current training configuration uses the four selected emotion classes:

```text
ANG → Angry
HAP → Happy
NEU → Neutral
SAD → Sad
```

---



## Deployment

The application can be deployed as a FastAPI web service.

For Render, use:

**Build Command**

```bash
pip install -r requirements.txt
```

**Start Command**

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

A Python version compatible with the TensorFlow dependencies should be selected for the deployment environment.

---



## Author

**Shiv Modi** — B.Tech. + M.Tech., IIT Bombay  
[GitHub](https://github.com/shivmodi21) · [Portfolio](https://shivmodi21.github.io/) · [LinkedIn](https://www.linkedin.com/in/shivmodi210/)