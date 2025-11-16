🧠 FETA – Facial Emotion, Age & Gender Recognition System

FETA (Facial Expression, Trait & Attribute Analyzer) is a deep learning–powered computer vision system designed to detect:

🎭 Facial Emotion (FER2013)

Angry

Disgust

Fear

Happy

Sad

Surprise

Neutral

👤 Age Prediction (UTKFace)

Predicts approximate human age based on facial features.

🚻 Gender Classification (UTKFace)

Classifies gender into Male / Female.

📌 Summary

FETA is an AI-based real-time facial analysis system built using TensorFlow/Keras, OpenCV, and pretrained CNN models. The system supports both live webcam inference and image-based input and includes full training and evaluation pipelines for reproduction and experimentation.

It is designed for:

Human–computer interaction

Emotion-aware systems

Behavioral analytics

Research & educational purposes

🔑 Key Features

✔ Real-time video inference with webcam
✔ Emotion + Age + Gender predictions
✔ Custom-trained deep learning models
✔ Transfer learning on UTKFace & FER2013
✔ Model evaluation tools included
✔ Training scripts provided
✔ Efficient preprocessing pipeline
✔ Modular architecture (easy to expand)

🧬 Model Details
Task	Dataset	Framework	Input Size
Emotion Classification	FER2013	Keras CNN	48×48 grayscale
Age & Gender Prediction	UTKFace	Transfer Learning	64×64 RGB

Training enhancements include:

One-hot encoded labels

Data augmentation

Early stopping & LR scheduling

Custom loss weighting for class imbalance

Improved model iteration (train_age_gender_improved.py)

🧰 Tech Stack

Python

TensorFlow / Keras

NumPy / Pandas

OpenCV

Matplotlib

Jupyter (optional)

CUDA / GPU support (optional)

📂 Example Project Structure
FETA/
│
├── app.py                    # optional Flask UI
├── camera.py                 # real-time webcam detection
├── train_age_gender_improved.py
├── train_emotion_keras.py
├── evaluate_emotion.py
├── evaluate_age_gender.py
│
├── models/                   # saved models (ignored for Git)
├── archive/                  # raw training data
├── processed_data/           # preprocessed numpy arrays
├── UTKFace/                  # dataset folder (ignored)
│
├── requirements.txt
└── README.md

🖥️ Usage
Run webcam detection
python camera.py

Evaluate an image
python emotion_detection.py --image input.jpg

📊 Training Performance
Model	Accuracy (approx.)
Emotion (FER2013)	70–75%
Gender	90%+
Age	±4–6 years error
⚠ Disclaimer

This project is for research and educational purposes only.
It is not intended for surveillance, psychological analysis, or real-world medical use.
