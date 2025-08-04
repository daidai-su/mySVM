import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import os # ファイルの存在確認のためにosライブラリをインポート

# --- 定数 ---
CANVAS_SIZE = 200
PEN_RADIUS = 12
DATA_FILE = 'handwritten_data.npz' # 保存するファイル名

# --- GUIアプリ本体 ---
class App:
    def __init__(self, master):
        self.master = master
        master.title("Handwritten Digit Classifier (w/ Save & Load)")

        # --- データ保存用変数 ---
        self.data_X = []
        self.data_y = []
        self.model = None

        # --- UI要素の配置 ---
        # ... (UIの配置コードは前回と同じなので省略) ...
        self.canvas = tk.Canvas(master, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white', relief=tk.RIDGE, bd=2)
        self.canvas.grid(row=0, column=0, columnspan=3, pady=10, padx=10)
        tk.Label(master, text="Correct Digit:", font=('Arial', 12)).grid(row=1, column=0, sticky=tk.E)
        self.entry = tk.Entry(master, width=5, font=('Arial', 14))
        self.entry.grid(row=1, column=1, sticky=tk.W)
        self.save_button = tk.Button(master, text="Save This Image", command=self.save_digit, bg='#4CAF50', fg='white', font=('Arial', 10))
        self.save_button.grid(row=1, column=2, padx=5, sticky=tk.W+tk.E)
        self.predict_button = tk.Button(master, text="Predict", command=self.predict, font=('Arial', 12, 'bold'))
        self.predict_button.grid(row=3, column=0, pady=10, sticky=tk.W+tk.E)
        self.clear_button = tk.Button(master, text="Clear", command=self.clear, font=('Arial', 12))
        self.clear_button.grid(row=3, column=1, columnspan=2, pady=10, sticky=tk.W+tk.E)
        self.train_button = tk.Button(master, text="Train on Collected Data", command=self.train_model, bg='#2196F3', fg='white', font=('Arial', 12, 'bold'))
        self.train_button.grid(row=4, column=0, columnspan=3, pady=5, sticky=tk.W+tk.E)
        self.label = tk.Label(master, text="Starting up...", font=('Arial', 14))
        self.label.grid(row=2, column=0, columnspan=3, pady=10)
        
        # 描画用イメージの初期化
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

        # ★★★ アプリ起動時にデータを読み込む ★★★
        self.load_data_from_file()
        
        # ★★★ ウィンドウを閉じる時にデータを保存する設定 ★★★
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def paint(self, event):
        # ... (前回と同じ) ...
        x1, y1 = (event.x - PEN_RADIUS), (event.y - PEN_RADIUS)
        x2, y2 = (event.x + PEN_RADIUS), (event.y + PEN_RADIUS)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        # ... (前回と同じ) ...
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="1. Draw -> 2. Enter Label -> 3. Save")
        self.entry.delete(0, tk.END)

    def _preprocess_image(self):
        # ... (前回と同じ) ...
        bbox = self.image.getbbox()
        if bbox is None: return None
        img_cropped = self.image.crop(bbox)
        width, height = img_cropped.size
        new_size = max(width, height)
        new_img = Image.new("L", (new_size, new_size), color=255)
        paste_x = (new_size - width) // 2
        paste_y = (new_size - height) // 2
        new_img.paste(img_cropped, (paste_x, paste_y))
        img_resized = new_img.resize((8, 8), Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=0.5))
        img_inverted = ImageOps.invert(img_resized)
        data = np.array(img_inverted).astype(np.float32)
        data = (data / 255.0) * 16.0
        data[data < 2.0] = 0.0
        return data.reshape(1, -1)

    def save_digit(self):
        # ... (前回と同じ) ...
        label_text = self.entry.get()
        if not label_text.isdigit():
            messagebox.showerror("Error", "Please enter a valid digit (0-9).")
            return
        label = int(label_text)
        data = self._preprocess_image()
        if data is None:
            messagebox.showerror("Error", "The canvas is empty.")
            return
        self.data_X.append(data[0])
        self.data_y.append(label)
        self.label.config(text=f"Saved '{label}'. Total samples: {len(self.data_y)}")
        self.clear()

    def train_model(self):
        # ... (前回と同じ) ...
        if len(self.data_y) < 20:
            messagebox.showinfo("Info", f"Not enough data ({len(self.data_y)} samples).\nPlease collect at least 20 samples.")
            return
        X_train = np.array(self.data_X)
        y_train = np.array(self.data_y)
        self.label.config(text="Training... (Please wait)")
        self.master.update()
        param_grid = {'svc__C': [10, 100, 1000], 'svc__gamma': [0.1, 0.01, 0.001]}
        pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
        grid_search = GridSearchCV(pipeline, param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        messagebox.showinfo("Success", "Your custom model has been trained!")
        self.label.config(text="Training complete! Ready to predict.")

    def predict(self):
        # ... (前回と同じ) ...
        if self.model is None:
            messagebox.showerror("Error", "The model has not been trained yet.\nPlease collect data and press 'Train'.")
            return
        data = self._preprocess_image()
        if data is None: return
        pred = self.model.predict(data)[0]
        self.label.config(text=f"Prediction: {pred}", font=('Arial', 24, 'bold'))
        
    # ★★★ 以下、新しく追加・修正した関数 ★★★
    
    def load_data_from_file(self):
        """起動時にファイルからデータを読み込む"""
        if os.path.exists(DATA_FILE):
            try:
                with np.load(DATA_FILE) as data:
                    self.data_X = list(data['X'])
                    self.data_y = list(data['y'])
                self.label.config(text=f"Loaded {len(self.data_y)} samples from file.\nReady to Train or add more data.")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load data file: {e}")
        else:
            self.label.config(text="No data file found. Please create new data.")

    def on_closing(self):
        """ウィンドウを閉じる時にデータをファイルに保存する"""
        if len(self.data_y) > 0:
            try:
                X_arr = np.array(self.data_X)
                y_arr = np.array(self.data_y)
                # 圧縮して保存
                np.savez_compressed(DATA_FILE, X=X_arr, y=y_arr)
                print(f"Successfully saved {len(y_arr)} samples to {DATA_FILE}")
            except Exception as e:
                print(f"Error saving data: {e}")
        
        self.master.destroy()

# --- アプリの起動 ---
root = tk.Tk()
app = App(root)
root.mainloop()
