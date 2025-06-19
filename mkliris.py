import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from MKLpy.metrics import pairwise
from MKLpy.algorithms import EasyMKL

# データ読み込み・分割・スケーリング
X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_te = scaler.transform(X_te)

# カーネル群の用意 (例: RBFカーネル 5種類)
gamma_list = np.logspace(-2, 2, 5)  # γ = 0.01, 0.1, 1, 10, 100
K_list = [pairwise.rbf_kernel(X_tr, gamma=g) for g in gamma_list]

# EasyMKL実行
mkl = EasyMKL(lam=0.1).fit(K_list, y_tr)

# 学習済みカーネルの重み確認
print("Learned kernel weights:", mkl.solution)

# テスト用カーネル行列も同様に作成
K_list_test = [pairwise.rbf_kernel(X_te, X_tr, gamma=g) for g in gamma_list]

# 予測
y_pred = mkl.predict(K_list_test)

# 評価
acc_train = accuracy_score(y_tr, mkl.predict(K_list))
acc_test  = accuracy_score(y_te, y_pred)
print(f"Train accuracy: {acc_train:.3f}  Test accuracy: {acc_test:.3f}")
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from MKLpy.metrics import pairwise
from MKLpy.algorithms import EasyMKL

# データ読み込み・分割・スケーリング
X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_te = scaler.transform(X_te)

# カーネル群の用意 (例: RBFカーネル 5種類)
gamma_list = np.logspace(-2, 2, 5)  # γ = 0.01, 0.1, 1, 10, 100
K_list = [pairwise.rbf_kernel(X_tr, gamma=g) for g in gamma_list]

# EasyMKL実行
mkl = EasyMKL(lam=0.1).fit(K_list, y_tr)

# 学習済みカーネルの重み確認
print("Learned kernel weights:", mkl.solution)

# テスト用カーネル行列も同様に作成
K_list_test = [pairwise.rbf_kernel(X_te, X_tr, gamma=g) for g in gamma_list]

# 予測
y_pred = mkl.predict(K_list_test)

# 評価
acc_train = accuracy_score(y_tr, mkl.predict(K_list))
acc_test  = accuracy_score(y_te, y_pred)
print(f"Train accuracy: {acc_train:.3f}  Test accuracy: {acc_test:.3f}")
for cls, sol in mkl.solution.items():
    print(f"Class {cls}: kernel weights = {sol.weights}")
