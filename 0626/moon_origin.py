import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# データ生成
X, y = make_moons(n_samples=300, noise=0, random_state=42)

# 散布図描画
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolors='k')
plt.title("Moon Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
