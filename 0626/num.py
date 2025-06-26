import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. データの読み込み
digits = load_digits()
X = digits.data  # (1797, 64) = 8x8画像を1次元ベクトルにしたもの
y = digits.target  # ラベル（0〜9）

# 2. スケーリング（重要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 訓練・テスト分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. SVM 学習（RBFカーネル）
clf = SVC(kernel='rbf', gamma=0.05, C=1)
clf.fit(X_train, y_train)

# 5. 評価
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 以下を分類コードの直後に追加してください
from sklearn.decomposition import PCA

# PCAで2次元に圧縮（可視化用）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA空間で分類モデルを再学習（ここでは全データ使用）
clf_2d = SVC(kernel='rbf', gamma=0.05, C=1)
clf_2d.fit(X_pca, y)

# グラフ表示
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.tab10, s=15, edgecolors='k')
plt.title("Digits dataset PCA + SVM(RBF) classification")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(*scatter.legend_elements(), title="Digits")
plt.show()
