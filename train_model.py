import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# Load CSV files
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

# Split features and labels
x_train = train_df.iloc[:, 1:].values / 255.0  # Normalize
y_train = train_df.iloc[:, 0].values

x_test = test_df.iloc[:, 1:].values / 255.0
y_test = test_df.iloc[:, 0].values

# Optional: One-hot encode labels
lb = LabelBinarizer()
y_train_encoded = lb.fit_transform(y_train)
y_test_encoded = lb.transform(y_test)

# Build a neural network
model = Sequential([
    Dense(128, input_shape=(784,), activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train_encoded, epochs=10, batch_size=64, validation_split=0.1)

# ✅ Save model
model.save("mnist_model.h5")
print("Model saved as mnist_model.h5")

# Evaluate
loss, acc = model.evaluate(x_test, y_test_encoded)
print(f"\nTest Accuracy: {acc:.2f}")

# Predict and print classification report
y_pred = np.argmax(model.predict(x_test), axis=1)
print(classification_report(y_test, y_pred))
