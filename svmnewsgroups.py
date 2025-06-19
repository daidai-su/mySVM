from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# データ取得（vectorized 版は TF-IDF 付き、疎行列で返る）
train = fetch_20newsgroups_vectorized(subset="train", remove=("headers", "footers", "quotes"))
test  = fetch_20newsgroups_vectorized(subset="test", remove=("headers", "footers", "quotes"))

X_tr, y_tr = train.data, train.target
X_te, y_te = test.data,  test.target

# 疎行列なので StandardScaler はスキップ（非線形SVMは要注意）
kernel_params = {
    "linear" : dict(kernel="linear"),
    "poly"   : dict(kernel="poly", degree=3, gamma="scale", coef0=0),
    "rbf"    : dict(kernel="rbf",  gamma="scale"),
    "sigmoid": dict(kernel="sigmoid", gamma="scale", coef0=0),
}

print("SVM comparison on 20newsgroups\n")
for name, params in kernel_params.items():
    clf = SVC(C=1.0, **params)
    clf.fit(X_tr, y_tr)
    tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
    te_acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"{name:<7}  train={tr_acc:.3f}  test={te_acc:.3f}")
