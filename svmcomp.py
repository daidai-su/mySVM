import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, make_moons, fetch_openml, fetch_20newsgroups_vectorized
from sklearn.utils import resample
from sklearn.decomposition import PCA
from scipy.sparse import vstack

def svm_and_report(X, y, dataset_name, kernels):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # 疎行列なら dense 変換（非線形カーネル時だけ）
    dense_needed = any(k["kernel"] != "linear" for k in kernels.values())

    if hasattr(X_tr, "toarray") and dense_needed:
        X_tr = X_tr.toarray()
        X_te = X_te.toarray()

    if dense_needed:
        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_te = scaler.transform(X_te)

    print(f"\nSVM comparison on {dataset_name}")
    print("-" * 50)
    for name, params in kernels.items():
        clf = SVC(C=1.0, **params)
        clf.fit(X_tr, y_tr)
        tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
        te_acc = accuracy_score(y_te, clf.predict(X_te))
        print(f"{name:<7}  train={tr_acc:.3f}  test={te_acc:.3f}")

# ===========================================
# ① Iris
iris = load_iris()
svm_and_report(
    iris.data, iris.target, "iris",
    kernels={
        "linear": dict(kernel="linear"),
        "poly": dict(kernel="poly", degree=3, gamma="scale", coef0=0),
        "rbf": dict(kernel="rbf", gamma="scale"),
        "sigmoid": dict(kernel="sigmoid", gamma="scale", coef0=0),
    }
)

# ② make_moons
X_moons, y_moons = make_moons(n_samples=1500, noise=0.25, random_state=0)
svm_and_report(
    X_moons, y_moons, "make_moons",
    kernels={
        "linear": dict(kernel="linear"),
        "poly": dict(kernel="poly", degree=3, gamma="scale", coef0=0),
        "rbf": dict(kernel="rbf", gamma="scale"),
        "sigmoid": dict(kernel="sigmoid", gamma="scale", coef0=0),
    }
)

# ③ MNIST (省エネ: サンプル減＋PCA次元削減)
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X_mnist = mnist.data / 255.0
y_mnist = mnist.target.astype(int)

X_small, y_small = resample(X_mnist, y_mnist, n_samples=5000, stratify=y_mnist, random_state=42)
pca = PCA(n_components=50).fit(X_small)
X_small_pca = pca.transform(X_small)

svm_and_report(
    X_small_pca, y_small, "MNIST (PCA 50)", 
    kernels={
        "linear": dict(kernel="linear"),
        "rbf": dict(kernel="rbf", gamma="scale"),
    }
)

# ④ 20newsgroups (疎行列のまま、サンプル数削減、linearのみ)
train = fetch_20newsgroups_vectorized(subset="train", remove=("headers", "footers", "quotes"))
test  = fetch_20newsgroups_vectorized(subset="test",  remove=("headers", "footers", "quotes"))

# サンプル数減らしてから疎行列で結合
X_train_small, y_train_small = resample(train.data, train.target, n_samples=3000, stratify=train.target, random_state=42)
X_test_small,  y_test_small  = resample(test.data,  test.target,  n_samples=1000, stratify=test.target, random_state=42)

X_all = vstack([X_train_small, X_test_small])
y_all = np.hstack([y_train_small, y_test_small])

svm_and_report(
    X_all, y_all, "20newsgroups (sparse)", 
    kernels={
        "linear": dict(kernel="linear"),
    }
)
