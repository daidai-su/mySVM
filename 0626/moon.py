import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# データ生成
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

# 訓練・テスト分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# スケーリング（SVMに重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4種類のSVMモデル
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
models = []

# SVC関数に各カーネルを指定するだけで勝手に計算してくれる
for k in kernels:
    clf = SVC(kernel=k, gamma='auto', probability=True)
    # ここで、モデルに 訓練データを使って学習させる処理。
    # これでwやｂのような値がきまる
    clf.fit(X_train_scaled, y_train)
    # (カーネル名, 訓練済みモデル) をタプルにしてリストに追加
    models.append((k, clf))

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

plt.figure(figsize=(12, 10))

for i, (name, clf) in enumerate(models):
    plt.subplot(2, 2, i + 1)
    plot_decision_boundary(clf, X_test_scaled, y_test, f"SVM kernel = {name}")

plt.tight_layout()
plt.show()
