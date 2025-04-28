import numpy as np
import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw
import tensorflow as tf

"""
Ниже расписан класс для распознавания цифр. 
"""


class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание цифр")
        
        self.canvas = Canvas(root, width=280, height=280, bg='black')
        self.canvas.pack()
        
        # Кнопки
        self.btn_recognize = Button(root, text="Распознать", command=self.recognize)
        self.btn_recognize.pack(side=tk.LEFT)
        
        self.btn_clear = Button(root, text="Очистить", command=self.clear)
        self.btn_clear.pack(side=tk.RIGHT)
        
        self.image = Image.new('L', (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.model = self.train_model()
    
    def train_model(self):
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        
        model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
        model.save('mnist_cnn.h5')
        print("Модель обучена и сохранена!")
        return model
    
    def paint(self, event):
        x, y = event.x, event.y
        r = 10 
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
        
    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        
    def recognize(self):
        img = self.image.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        print(f"Это цифра: {digit} (уверенность: {confidence*100:.1f}%)")
        self.root.title(f"Результат: {digit} | Уверенность: {confidence*100:.1f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()