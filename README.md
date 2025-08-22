📝 Handwritten Digit Recognizer
📌 Project Overview

Manual recognition of handwritten digits is slow and error-prone, especially with large volumes of data (e.g., bank cheques, forms).
This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to automatically recognize handwritten digits (0–9) from the MNIST dataset with high accuracy.

🚀 Problem Statement

How can we build a deep learning model using TensorFlow that accurately recognizes handwritten digits from images?

🎯 Objectives

Preprocess handwritten digit images.

Build a CNN model for classification.

Train & evaluate the model on MNIST dataset.

Predict digits from unseen test images.

Visualize results and CNN architecture.

📂 Project Workflow

Dataset Loading – MNIST dataset from TensorFlow/Keras.

Preprocessing – Normalize pixel values (0–1), reshape images to (28×28×1).

Model Building – CNN with Conv2D, MaxPooling2D, Flatten, Dense layers.

Training – Train using training set, validate on test set.

Evaluation – Accuracy and loss metrics.

Prediction & Visualization – Display predictions for sample digits.

Model Visualization – CNN architecture diagram + training curves.

📊 Dataset Details

Source: MNIST (Keras built-in).

Shape: 60,000 training images, 10,000 test images.

Image size: 28 × 28 grayscale.

Labels: Digits 0–9.

Pixel values: 0–255, normalized to 0–1.

🏗️ Model Architecture (CNN)

Conv2D(32 filters, 3×3, ReLU)

MaxPooling2D(2×2)

Conv2D(64 filters, 3×3, ReLU)

MaxPooling2D(2×2)

Flatten

Dense(128, ReLU)

Dense(10, Softmax)

🛠️ Tech Stack & Requirements
Languages & Libraries

Python 3.x

TensorFlow / Keras

NumPy

Matplotlib

Installation
pip install tensorflow numpy matplotlib

🖥️ How to Run

Clone this repository

git clone https://github.com/your-username/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer


Run the training script

python mnist_cnn.py


Expected Output

Model trains to ~98% accuracy on test set.

Displays sample predictions.

Saves CNN architecture diagram as cnn_architecture.png.

📸 Visualizations
🔹 Sample MNIST Digits

Example of handwritten digits used for training.

🔹 CNN Architecture Diagram

Generated using plot_model() from Keras utilities.

🔹 Training Curves

Graph of accuracy & loss vs. epochs.

✅ Results

Achieved ~98% accuracy on test data.

Model generalizes well to unseen digit images.

📌 Future Work

Deploy model using Flask / FastAPI / Streamlit.

Convert to TensorFlow Lite for mobile apps.

Extend to recognizing handwritten characters (A–Z).

🙌 Acknowledgements

MNIST dataset: Yann LeCun et al.

TensorFlow/Keras for deep learning framework.
