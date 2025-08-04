import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import rbf_kernel
import os

# --- 定数と設定 -----------------------------------------------------------
CANVAS_SIZE = 200
PEN_RADIUS = 12
DATA_FILE = 'handwritten_data.npz'

# ★★★ ここで使用するカーネルを選択 ★★★
# 'rbf', 'linear', 'poly', 'mkl_rbf' から選んでください
SELECTED_KERNEL = 'linear'
# --------------------------------------------------------------------------

# --- カーネルごとの設定を定義 ---
def multi_scale_rbf_kernel(X1, X2):
    """MKL(RBF)のためのカスタムカーネル関数"""
    gammas = np.logspace(-3, 2, 20) # スケール数を調整
    kernel_sum = np.zeros((X1.shape[0], X2.shape[0]))
    for gamma in gammas:
        kernel_sum += rbf_kernel(X1, X2, gamma=gamma)
    return kernel_sum

# 各カーネルのモデルとチューニング用パラメータを辞書で管理
KERNEL_CONFIGS = {
    'rbf': {
        'model': SVC(kernel='rbf', probability=True),
        'param_grid': {'svc__C': [10, 100, 500], 'svc__gamma': [0.1, 0.01, 0.001]}
    },
    'linear': {
        'model': SVC(kernel='linear', probability=True),
        'param_grid': {'svc__C': [0.1, 1, 10, 100]}
    },
    'poly': {
        'model': SVC(kernel='poly', probability=True),
        'param_grid': {'svc__C': [10, 100], 'svc__degree': [3, 4], 'svc__gamma': ['scale', 'auto']}
    },
    'mkl_rbf': {
        'model': SVC(kernel=multi_scale_rbf_kernel, probability=True),
        'param_grid': {'svc__C': [10, 100, 500, 1000]}
    }
}

# --- GUIアプリ本体 ---
class App:
    def __init__(self, master):
        self.master = master
        # ウィンドウタイトルに選択中のカーネル名を表示
        master.title(f"Digit Classifier - Kernel: {SELECTED_KERNEL.upper()}")

        self.data_X = []
        self.data_y = []
        self.model = None

        # --- UI要素の配置 (変更なし) ---
        self.canvas = tk.Canvas(master, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white', relief=tk.RIDGE, bd=2)
        self.canvas.grid(row=0, column=0, columnspan=3, pady=10, padx=10)
        tk.Label(master, text="Correct Digit:", font=('Arial', 12)).grid(row=1, column=0, sticky=tk.E)
        self.entry = tk.Entry(master, width=5, font=('Arial', 14))
        self.entry.grid(row=1, column=1, sticky=tk.W)
        self.save_button = tk.Button(master, text="Save", command=self.save_digit, bg='#4CAF50', fg='white', font=('Arial', 10))
        self.save_button.grid(row=1, column=2, padx=5, sticky=tk.W+tk.E)
        self.predict_button = tk.Button(master, text="Predict", command=self.predict, font=('Arial', 12))
        self.predict_button.grid(row=3, column=0, pady=10, sticky=tk.W+tk.E)
        self.clear_button = tk.Button(master, text="Clear", command=self.clear, font=('Arial', 12))
        self.clear_button.grid(row=3, column=1, columnspan=2, pady=10, sticky=tk.W+tk.E)
        self.train_button = tk.Button(master, text=f"Train with '{SELECTED_KERNEL}' kernel", command=self.train_model, font=('Arial', 12, 'bold'))
        self.train_button.grid(row=4, column=0, columnspan=3, pady=5, sticky=tk.W+tk.E)
        self.label = tk.Label(master, text="Starting up...", font=('Arial', 14))
        self.label.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.load_data_from_file()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def train_model(self):
        """保存したデータで、選択されたカーネルのSVMモデルを学習する"""
        if len(self.data_y) < 20:
            messagebox.showinfo("Info", f"Not enough data ({len(self.data_y)} samples).\nPlease collect at least 20 samples.")
            return

        # 選択されたカーネルの設定を取得
        if SELECTED_KERNEL not in KERNEL_CONFIGS:
            messagebox.showerror("Error", f"Invalid kernel name: '{SELECTED_KERNEL}'")
            return
            
        config = KERNEL_CONFIGS[SELECTED_KERNEL]
        
        X_train = np.array(self.data_X)
        y_train = np.array(self.data_y)
        
        self.label.config(text=f"Training with '{SELECTED_KERNEL}'... (Please wait)")
        self.master.update()

        # パイプラインとGridSearchCVを、選択されたカーネルの設定で構築
        pipeline = make_pipeline(StandardScaler(), config['model'])
        grid_search = GridSearchCV(pipeline, config['param_grid'], cv=3, n_jobs=-1) # n_jobs=-1で高速化
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        messagebox.showinfo("Success", f"Custom '{SELECTED_KERNEL}' model has been trained!")
        self.label.config(text="Training complete! Ready to predict.")
        
    # --- 他の関数 (paint, clear, _preprocess_image, save_digit, predict, load_data_from_file, on_closing) は変更なし ---
    def paint(self, event):
        x1, y1 = (event.x - PEN_RADIUS), (event.y - PEN_RADIUS)
        x2, y2 = (event.x + PEN_RADIUS), (event.y + PEN_RADIUS)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        #self.label.config(text="1. Draw -> 2. Enter Label -> 3. Save")
        self.entry.delete(0, tk.END)

    def _preprocess_image(self):
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

    def predict(self):
        if self.model is None:
            messagebox.showerror("Error", "The model has not been trained yet.\nPlease collect data and press 'Train'.")
            return
        data = self._preprocess_image()
        if data is None: return
        pred = self.model.predict(data)[0]
        self.label.config(text=f"Prediction: {pred}", font=('Arial', 24, 'bold'))
        
    def load_data_from_file(self):
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
        if len(self.data_y) > 0:
            try:
                np.savez_compressed(DATA_FILE, X=np.array(self.data_X), y=np.array(self.data_y))
                print(f"Successfully saved {len(self.data_y)} samples to {DATA_FILE}")
            except Exception as e:
                print(f"Error saving data: {e}")
        self.master.destroy()

# --- アプリの起動 ---
if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
