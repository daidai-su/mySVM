import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 手書きキャンバスの設定
canvas_size = 200  # キャンバスは200x200ピクセル

# SVMモデルの訓練
digits = load_digits()
X, y = digits.data, digits.target
model = make_pipeline(StandardScaler(), SVC(gamma=0.001))
model.fit(X, y)

# GUIアプリ本体
class App:
    def __init__(self, master):
        self.master = master
        master.title("手書き数字分類 (SVM)")

        self.canvas = tk.Canvas(master, width=canvas_size, height=canvas_size, bg='white')
        self.canvas.pack()

        self.label = tk.Label(master, text="描いて → 『予測』を押す", font=('Arial', 16))
        self.label.pack()

        self.predict_button = tk.Button(master, text="予測", command=self.predict)
        self.predict_button.pack()

        self.clear_button = tk.Button(master, text="クリア", command=self.clear)
        self.clear_button.pack()

        self.image = Image.new("L", (canvas_size, canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        r = 8  # 筆の太さ
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (canvas_size, canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="描いて → 『予測』を押す")

    def predict(self):
        img_resized = self.image.resize((8, 8), Image.LANCZOS)
        img_resized = ImageOps.invert(img_resized)
        data = np.array(img_resized).astype(np.float32)
        data = data / 16.0  # SVMの入力に合わせる
        data = data.reshape(1, -1)
        pred = model.predict(data)[0]
        self.label.config(text=f"予測: {pred}", font=('Arial', 18))

root = tk.Tk()
app = App(root)
root.mainloop()
