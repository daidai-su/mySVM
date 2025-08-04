import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel

# データ生成
X, y = make_moons(n_samples=300, noise=0, random_state=42)

# 訓練・テスト分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# スケーリング（SVMに重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4種類のSVMモデル（単体）
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
models = []

for k in kernels:
    clf = SVC(kernel=k, gamma='auto', probability=True)
    clf.fit(X_train_scaled, y_train)
    models.append((k, clf))

# 🔶 重み付き平均カーネルの定義
def weighted_kernel(X, Y):
    w1, w2, w3, w4 = 0.99, 0.005, 0.0025, 0.0025  # 重みは自由に調整可能
    K1 = linear_kernel(X, Y)
    K2 = polynomial_kernel(X, Y, degree=10)
    K3 = rbf_kernel(X, Y, gamma=1.0)
    K4 = sigmoid_kernel(X, Y)
    return w1*K1 + w2*K2 + w3*K3 + w4*K4

# 重み付き平均カーネルで学習
weighted_clf = SVC(kernel=weighted_kernel)
weighted_clf.fit(X_train_scaled, y_train)
models.append(("weighted", weighted_clf))

# 決定境界を描画する関数
def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')

# プロット（2行3列で5つ表示）
plt.figure(figsize=(15, 10))

for i, (name, clf) in enumerate(models):
    plt.subplot(2, 3, i + 1)
    plot_decision_boundary(clf, X_test_scaled, y_test, f"SVM kernel = {name}")

plt.tight_layout()
plt.show()
