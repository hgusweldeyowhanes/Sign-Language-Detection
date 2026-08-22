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
- **CNN-Based Model:** Deep learning model for accurate classification.  
- **Multiple Classes:** Supports customizable sign language gestures.  
- **Preprocessing Pipeline:** Includes hand tracking and image preprocessing.  
- **Performance Metrics:** Evaluated using accuracy, confusion matrix, and balanced accuracy.  
- **Conversation Mode (advanced):** Stable signs become a live sentence, with TTS speak-back, guided practice targets, confidence meters, undo/clear, and TXT/JSON transcript export (`python run_advanced.py`).  

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
4. Place your trained weights at the project root as `best_model.keras`.
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
```
Features: live confidence bars, sentence buffer, text-to-speech, practice targets, export transcript.

### Classic notebook UI
```bash
jupyter notebook Prediction.ipynb
```
## Future Improvements

Expand dataset with more sign language gestures

Add multilingual sign language support

Deploy as a web or mobile app
## Conclusion

