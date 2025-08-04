import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

# --------------------------------------------------------------------------
# Step 1: 複雑なスイスロールデータセットを生成する関数
# --------------------------------------------------------------------------
def make_complex_swiss_roll(n_samples=1000, n_turns=4, n_stripes=16, noise=0.1):
    """
    クラスが細かく入れ替わる、分類問題としてのスイスロールを生成する。
    """
    np.random.seed(42)
    
    # 螺旋の角度と高さを生成
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    y = 21 * np.random.rand(n_samples)
    
    # スイスロールの3D座標を計算
    x_coords = t * np.cos(t)
    z_coords = t * np.sin(t)
    
    X = np.vstack((x_coords, y, z_coords)).T
    
    # ロールに沿ってクラスを細かく交互に割り当てる
    # 角度tを基準に、n_stripes回クラスが入れ替わるようにする
    labels = np.floor(t * n_stripes / (max(t))) % 2
    
    # ノイズの追加
    X += noise * np.random.randn(n_samples, 3)
    
    return X, labels, t

# --------------------------------------------------------------------------
# Step 2: 数十個のRBFカーネルを統合するカスタムカーネル
# --------------------------------------------------------------------------
def multi_scale_rbf_kernel(X1, X2):
    """
    対数スケールで配置した数十個の異なるgammaを持つRBFカーネルを足し合わせる。
    """
    # 0.001から1000まで、対数的に均等な30個のgamma値を生成
    gammas = np.logspace(-3, 3, 30)
    
    # 各gammaで計算したカーネル行列を格納するリスト
    kernel_sum = np.zeros((X1.shape[0], X2.shape[0]))
    
    # 全てのスケールのカーネルを足し合わせる
    for gamma in gammas:
        kernel_sum += rbf_kernel(X1, X2, gamma=gamma)
        
    return kernel_sum

# --------------------------------------------------------------------------
# Step 3: データの準備とモデルの定義
# --------------------------------------------------------------------------
X, y, t = make_complex_swiss_roll(n_samples=1500, n_turns=4, n_stripes=12, noise=0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 比較モデルの定義
models = {
    "Single RBF (gamma=0.1)": SVC(kernel='rbf', gamma=0.1, C=100),
    "MKL (30 Combined RBFs)": SVC(kernel=multi_scale_rbf_kernel, C=100)
}

# --------------------------------------------------------------------------
# Step 4: モデルの学習と3Dでの結果可視化
# --------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm)
ax1.set_title("Original Data (True Labels)")
ax1.view_init(elev=10, azim=-70)

# MKLモデルの学習と予測
mkl_model = models["MKL (30 Combined RBFs)"]
mkl_model.fit(X_train, y_train)
y_pred_mkl = mkl_model.predict(X)

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred_mkl, cmap=plt.cm.coolwarm)
acc_mkl = accuracy_score(y, y_pred_mkl)
ax2.set_title(f"MKL Prediction (Accuracy: {acc_mkl:.2f})")
ax2.view_init(elev=10, azim=-70)

plt.suptitle("3D Classification of Complex Swiss Roll", fontsize=16)
plt.show()

# --------------------------------------------------------------------------
# Step 5: 「展開図」での決定境界の可視化
# --------------------------------------------------------------------------
# 3Dデータを2Dの「展開図」に変換（t: 巻きつき角度, y: 高さ）
X_unrolled = np.vstack((t, X[:, 1])).T
X_train_unrolled, _, y_train_unrolled, _ = train_test_split(X_unrolled, y, test_size=0.3, random_state=42)

def plot_unrolled_boundary(model, ax, title):
    # 2Dの展開図データでモデルを再学習
    model.fit(X_train_unrolled, y_train_unrolled)
    
    h = .1
    x_min, x_max = X_unrolled[:, 0].min() - 1, X_unrolled[:, 0].max() + 1
    y_min, y_max = X_unrolled[:, 1].min() - 1, X_unrolled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X_unrolled[:, 0], X_unrolled[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=20)
    ax.set_title(title)
    ax.set_xlabel("Angle (Unrolled Axis)")
    ax.set_ylabel("Height")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_unrolled_boundary(models["Single RBF (gamma=0.1)"], axes[0], "Single RBF on Unrolled Data")
plot_unrolled_boundary(SVC(kernel=multi_scale_rbf_kernel, C=100), axes[1], "MKL on Unrolled Data")
plt.suptitle("Decision Boundary on 'Unrolled' 2D Data", fontsize=16)
plt.show()
