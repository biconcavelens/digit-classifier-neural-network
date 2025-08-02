import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# 🧠 Load your trained model
model = load_model("mnist_model.h5")

# 🖌️ GUI for drawing
class DigitClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draw a Digit")
        self.resizable(False, False)

        self.canvas = tk.Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack()

        self.image = Image.new("L", (280, 280), color="white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.draw_digit)

        tk.Button(self, text="Predict", command=self.predict).pack()
        tk.Button(self, text="Clear", command=self.clear_canvas).pack()

    def draw_digit(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill='white')

    def predict(self):
        # Convert canvas image to 28x28 like MNIST, invert color
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)  # white bg → black bg
        img_array = np.array(img).reshape(1, -1) / 255.0

        prediction = model.predict(img_array, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        self.title(f"Predicted: {digit} (Confidence: {confidence:.2f})")
        print(f"Prediction: {digit}, Confidence: {confidence:.2f}")

# 🔁 Run the app
if __name__ == "__main__":
    DigitClassifierApp().mainloop()
