# Sign Language Detection Using Deep Learning and Computer Vision

## Project Overview
This project aims to **detect and classify sign language gestures** using **Computer Vision** and a **Convolutional Neural Network (CNN)**.  
The model processes images or real-time webcam input to identify hand gestures, making it useful for accessibility, communication, and educational applications.

---
## Table of Contents
* [Features](#features)
* [Installation](#installation)
* [Dataset](#dataset)
* [Technologies-Used](#technologies-used)
* [Model-Architecture](#model-architecture)
* [Performance](#performance)
* [Usage](#usage)
* [Futere-Improvement](#future-improvements)
* [Conclusion](#conclusion)

## Features
- **Real-time Detection:** Live hand gesture recognition using a webcam.  
- **CNN-Based Model:** Deep learning model for accurate classification of phrase signs.  
- **Multiple Classes:** Supports customizable sign language gestures.  
- **Preprocessing Pipeline:** Includes hand tracking and image preprocessing.  
- **Performance Metrics:** Evaluated using accuracy, confusion matrix, and balanced accuracy.  
- **Conversation Mode (advanced):** Stable signs become a live sentence, with TTS speak-back, guided practice targets, confidence meters, undo/clear, and TXT/JSON transcript export (`python run_advanced.py`).  
- **Fingerspell A–Z:** Spell words letter by letter from MediaPipe landmarks (two-hand tracking, word buffer, pause-to-commit, double-letter bounce, practice words such as HELLO / HELP). Works without `best_model.keras`.  

---
## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hgusweldeyowhanes/Sign-Language-Detection.git
   cd Sign-Language-Detection
   ```
2. Create and activate a virtual environment:
 ```bash
 python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Phrase signs need trained weights at the project root as `best_model.keras`. **Fingerspell A–Z does not** — it uses the webcam and MediaPipe only.

## Dataset
The dataset consists of images of hand gestures representing different classes of sign language.
```bash
Data/
├── train/
│   ├── CLASS_1/
│   ├── CLASS_2/
│   └── CLASS_3/
├── test/
│   ├── CLASS_1/
│   ├── CLASS_2/
│   └── CLASS_3/
└── validation/
    ├── CLASS_1/
    ├── CLASS_2/
    └── CLASS_3/

```
## Technologies Used
- **Python 3.x**
- **OpenCV** – Image and video processing  
- **Mediapipe / cvzone** – Hand tracking & keypoint extraction  
- **TensorFlow / Keras** – CNN model training & inference  
- **NumPy & Pandas** – Data handling  
- **Matplotlib / Seaborn** – Performance visualization  

---


## Model Architecture
Input Layer: 224×224 RGB images

Convolutional Layers: Multiple Conv2D layers with ReLU activation

Pooling Layers: MaxPooling2D for dimensionality reduction

Dropout Layers: Prevents overfitting

Output Layer: Softmax classifier (number of gesture classes)

## Performance
Accuracy: ~95–100% (depending on dataset & training)

Confusion Matrix: Visualizes misclassifications

Balanced Accuracy: Effective with class imbalance

## Usage

### Advanced Conversation Studio (recommended)
```bash
python run_advanced.py
python run_advanced.py --mode fingerspell
```
Phrase mode: live confidence bars, sentence buffer, text-to-speech, practice targets, export transcript.

Fingerspell mode: A–Z from hand landmarks, current-word buffer, commit on pause, guided spelling practice (`HI`, `HELLO`, `HELP`, …). Hold a letter still to add it. Drop your hand briefly before repeating a letter (the two L’s in HELLO). J is static-I; Z is static-D because those letters need motion.

```bash
python -m pytest
```

### Classic notebook UI
```bash
jupyter notebook Prediction.ipynb
```
## Future Improvements

Train a landmark sequence model so J, Z, and other moving signs are recognized instead of approximated

Add multilingual sign language support

Deploy as a web or mobile app
## Conclusion

