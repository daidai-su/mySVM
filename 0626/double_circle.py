import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

# --------------------------------------------------------------------------
# Step 1: 既存のデータセットをロード
# --------------------------------------------------------------------------
# scikit-learnのmake_circlesを使用して、同心円状のデータセットを生成
# factor: 内側の円と外側の円の間の距離 (小さいほど難しい)
# noise: データ点のばらつき
X, y = datasets.make_circles(n_samples=400, factor=0.4, noise=0.1, random_state=42)

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# --------------------------------------------------------------------------
# Step 2: MKLのコンセプトを実装するカスタムカーネル関数
# --------------------------------------------------------------------------
def combined_mkl_kernel(X1, X2):
    """
    異なるgammaを持つ2つのRBFカーネルを足し合わせるカスタムカーネル。
    """
    # 局所的な特徴を捉えるカーネル（近視眼的）
    gamma_local = 10.0 
    k_local = rbf_kernel(X1, X2, gamma=gamma_local)
    
    # 大局的な特徴を捉えるカーネル（大雑把）
    gamma_global = 0.1
    k_global = rbf_kernel(X1, X2, gamma=gamma_global)
    
    # 2つのカーネルを足し合わせることで、両方の視点を統合する
    return k_local + k_global

# --------------------------------------------------------------------------
# Step 3: 比較モデルの定義
# --------------------------------------------------------------------------
models = {
    "RBF (Local, gamma=10.0)": SVC(kernel='rbf', gamma=10.0, C=10),
    "RBF (Global, gamma=0.1)": SVC(kernel='rbf', gamma=0.1, C=10),
    "MKL (Combined Kernel)": SVC(kernel=combined_mkl_kernel, C=10)
}

# --------------------------------------------------------------------------
# Step 4: モデルの学習と結果の可視化
# --------------------------------------------------------------------------
# 決定境界をプロットする関数
def plot_decision_boundary(model, ax, title):
    h = .05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

# 各モデルを学習・評価・プロット
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    title = f"{name}\nAccuracy: {acc:.2f}"
    plot_decision_boundary(model, axes[i], title)

plt.tight_layout()
plt.show()
