import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

# --------------------------------------------------------------------------
# Step 1: 二重螺旋データセットを生成する関数
# --------------------------------------------------------------------------
def make_spirals(n_samples, noise=0.5, n_turns=2):
    """
    二重螺旋のデータセットを生成します。
    """
    np.random.seed(42)
    
    n_per_class = n_samples // 2
    max_angle = n_turns * 2 * np.pi
    
    # クラス0の螺旋
    theta0 = np.sqrt(np.random.rand(n_per_class)) * max_angle
    r0 = theta0 + np.random.randn(n_per_class) * noise
    x0 = np.array([r0 * np.sin(theta0), r0 * np.cos(theta0)]).T
    
    # クラス1の螺旋
    theta1 = np.sqrt(np.random.rand(n_per_class)) * max_angle
    r1 = -theta1 - np.random.randn(n_per_class) * noise # 逆向きの螺旋
    x1 = np.array([r1 * np.sin(theta1), r1 * np.cos(theta1)]).T
    
    X = np.vstack([x0, x1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    return X, y

# --------------------------------------------------------------------------
# Step 2: MKLのコンセプトを実装するカスタムカーネル関数
# --------------------------------------------------------------------------
def combined_mkl_kernel(X1, X2):
    """
    異なるgammaを持つ2つのRBFカーネルを足し合わせるカスタムカーネル。
    """
    # 局所的な特徴を捉えるカーネル（近視眼的）
    gamma_local = 1.0 
    k_local = rbf_kernel(X1, X2, gamma=gamma_local)
    
    # 大局的な特徴を捉えるカーネル（大雑把）
    gamma_global = 0.01
    k_global = rbf_kernel(X1, X2, gamma=gamma_global)
    
    # 2つのカーネルを足し合わせることで、両方の視点を統合する
    return k_local + k_global

# --------------------------------------------------------------------------
# Step 3: データの準備とモデルの定義
# --------------------------------------------------------------------------
X, y = make_spirals(n_samples=400, noise=0.8, n_turns=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "RBF (Local, gamma=1.0)": SVC(kernel='rbf', gamma=1.0, C=10),
    "RBF (Global, gamma=0.01)": SVC(kernel='rbf', gamma=0.01, C=10),
    "MKL (Combined Kernel)": SVC(kernel=combined_mkl_kernel, C=10)
}

# --------------------------------------------------------------------------
# Step 4: モデルの学習と結果の可視化
# --------------------------------------------------------------------------
def plot_decision_boundary(model, ax, title):
    h = .1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    title = f"{name}\nAccuracy: {acc:.2f}"
    plot_decision_boundary(model, axes[i], title)

plt.tight_layout()
plt.show()
