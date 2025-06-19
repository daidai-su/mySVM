from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. データ読み込み
X, y = load_iris(return_X_y=True)

# 2. 学習 / テスト分割（クラス比を保つ）
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# 3. スケーリング（非線形カーネルは必須、linear でも揃えておくと無難）
scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_te = scaler.transform(X_te)

# 4. 基本 4 カーネル
kernel_params = {
    "linear" : dict(kernel="linear"),
    "poly"   : dict(kernel="poly", degree=3, gamma="scale", coef0=0),
    "rbf"    : dict(kernel="rbf",  gamma="scale"),
    "sigmoid": dict(kernel="sigmoid", gamma="scale", coef0=0),
}

print("SVM comparison on iris (train:test = 7:3)\n")
for name, params in kernel_params.items():
    clf = SVC(C=1.0, **params)
    clf.fit(X_tr, y_tr)
    tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
    te_acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"{name:<7}  train={tr_acc:.3f}  test={te_acc:.3f}")
