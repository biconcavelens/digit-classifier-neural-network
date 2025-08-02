# 🖌️ MNIST Digit Classifier with Drawing GUI

This project uses a neural network trained on the MNIST dataset to recognize handwritten digits. The user can draw a digit in the app's GUI and it will predict what digit the user wrote.

## 🚀 Features

- Simple neural network trained from CSV-based MNIST dataset
- Tkinter-based GUI where users can draw a digit
- Real-time prediction using trained model
- Model stored locally as `mnist_model.h5`

## 🧠 How It Works

1. **train_model.py** trains a neural network using `mnist_train.csv` and `mnist_test.csv`
2. The trained model is saved as `mnist_model.h5`
3. **gui_predictor.py** loads the model and opens a GUI canvas
4. You draw a digit, and it predicts the result

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
