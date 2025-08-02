
# 🖌️ MNIST Digit Classifier with Drawing GUI

This project uses a neural network trained on the MNIST dataset to recognize handwritten digits. You can draw a digit in the app's GUI and it will predict what digit you wrote — live!

(mnist_train.csv is larger than github upload limit, download from kaggle)

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
pip install tensorflow numpy pandas scikit-learn pillow
```

## 📂 File Descriptions

| File              | Description                            |
|-------------------|----------------------------------------|
| `train_model.py`  | Trains the neural network model        |
| `gui_predictor.py`| GUI app for drawing + prediction       |
| `mnist_train.csv` | Training data (download from Kaggle)   |
| `mnist_test.csv`  | Testing data (download from Kaggle)    |
| `requirements.txt`| Required Python packages               |

## 📊 Dataset

Download the CSV-formatted MNIST dataset from Kaggle:  
🔗 https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

## 🎮 Usage

1. Download the CSV files into the project folder.
2. Run the training script:

```bash
python train_model.py
```

3. Once trained, run the GUI:

```bash
python gui_predictor.py
```

Draw a digit, click **Predict**, and see the result!
