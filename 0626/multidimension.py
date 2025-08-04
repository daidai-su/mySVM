import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification, make_blobs
from sklearn.metrics.pairwise import rbf_kernel

# MKLpyから必要なクラスをインポート
try:
    from MKLpy.algorithms import MKL
except ImportError:
    print("MKLpy is not installed. Please run 'pip install mkl-py'")
    exit()

# --- Step 1: 5つの高次元データセット生成関数 (内容は省略) ---
def generate_high_dim_linear(n_samples=1000, n_features=5, random_state=42):
    return make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0, n_clusters_per_class=1, flip_y=0.01, class_sep=1.5, random_state=random_state)
def generate_high_dim_sphere(n_samples=1000, n_features=5, random_state=42):
    np.random.seed(random_state); X = np.random.randn(n_samples, n_features); distances = np.linalg.norm(X, axis=1); y = (distances > 2.5).astype(int); return X, y
def generate_high_dim_checkerboard(n_samples=1000, n_features=5, random_state=42):
    np.random.seed(random_state); X = np.random.uniform(-3, 3, (n_samples, n_features)); y = (np.floor(X).sum(axis=1) % 2).astype(int); return X, y
def generate_multi_scale_clusters(n_samples=1000, n_features=5, random_state=42):
    np.random.seed(random_state); X0, y0 = make_blobs(n_samples=n_samples//2, centers=[np.zeros(n_features)], cluster_std=1.5, n_features=n_features, random_state=random_state); n_satellites = 4; centers = []; [centers.append(c) for i in range(n_satellites) if (c := np.zeros(n_features), c.__setitem__(i % n_features, 5), True)[-1]]; X1, y1 = make_blobs(n_samples=n_samples//2, centers=centers, cluster_std=0.5, n_features=n_features, random_state=random_state); y1[:] = 1; X = np.vstack((X0, X1)); y = np.hstack((y0, y1)); return X, y
def generate_redundant_features(n_samples=1000, n_features=5, random_state=42):
    return make_classification(n_samples=n_samples, n_features=n_features, n_informative=2, n_redundant=3, n_clusters_per_class=1, flip_y=0.05, random_state=random_state)

# --- Step 2: カスタムカーネル関数と各種設定 ---

def simple_mkl_rbf_kernel(X1, X2):
    """単純平均MKLのためのカスタムカーネル関数"""
    gammas = np.logspace(-3, 2, 20)
    kernel_sum = np.zeros((X1.shape[0], X2.shape[0]))
    for gamma in gammas:
        kernel_sum += rbf_kernel(X1, X2, gamma=gamma)
    return kernel_sum / len(gammas) # 平均を取る

# 標準的なカーネルと単純MKLの設定
KERNEL_CONFIGS = {
    'Linear':          {'kernel': 'linear', 'C': 1.0},
    'Poly (d=4)':      {'kernel': 'poly', 'degree': 4, 'C': 10.0},
    'RBF (single)':    {'kernel': 'rbf', 'gamma': 'scale', 'C': 10.0},
    'Simple MKL-RBF':  {'kernel': simple_mkl_rbf_kernel, 'C': 100}
}

# --- Step 3: メイン処理 ---

if __name__ == '__main__':
    
    # --- ここで実験するデータセットを選択 ---
    # 'linear', 'sphere', 'checkerboard', 'multi_scale', 'redundant' から選んでください
    SELECTED_DATASET = 'multi_scale'
    # ------------------------------------

    dataset_generators = {
        'linear': generate_high_dim_linear, 'sphere': generate_high_dim_sphere,
        'checkerboard': generate_high_dim_checkerboard, 'multi_scale': generate_multi_scale_clusters,
        'redundant': generate_redundant_features
    }

    print(f"--- Starting experiment on '{SELECTED_DATASET}' dataset ---")

    X, y = dataset_generators[SELECTED_DATASET]()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data shape: Train={X_train.shape}, Test={X_test.shape}\n")
    
    # --- Part 1: 標準的なカーネルと単純MKLの評価 ---
    for name, params in KERNEL_CONFIGS.items():
        print(f"Training with kernel: {name}...")
        clf = SVC(**params)
        start_time = time.time()
        clf.fit(X_train_scaled, y_train)
        end_time = time.time()
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  -> Accuracy: {accuracy:.4f}, Time: {end_time - start_time:.2f}s\n")

    # --- Part 2: 最適な重み付けをしたMKL (MKLpy) の評価 ---
    print(f"Training with kernel: Weighted MKL-RBF (MKLpy)...")
    
    # 候補となるベースRBFカーネルのgamma値
    gammas = np.logspace(-3, 2, 20)
    
    # 各ベースカーネルでグラム行列を事前に計算
    KL_train = [rbf_kernel(X_train_scaled, gamma=g) for g in gammas]
    KL_test =  [rbf_kernel(X_test_scaled, X_train_scaled, gamma=g) for g in gammas]

    # MKL.SVCモデルを定義し、学習
    mkl = MKL.SVC(lam=0.1, C=100) # lamは重み学習の正則化パラメータ
    
    start_time = time.time()
    mkl.fit(KL_train, y_train)
    end_time = time.time()
    
    # 予測と評価
    y_pred = mkl.predict(KL_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  -> Accuracy: {accuracy:.4f}, Time: {end_time - start_time:.2f}s\n")

    # 学習されたカーネルの重み（theta）を可視化
    print("  -> Learned Kernel Weights (theta) by MKLpy:")
    top_weights = sorted(enumerate(mkl.theta), key=lambda x: x[1], reverse=True)[:5]
    for i, weight in top_weights:
        print(f"     - Kernel with gamma={gammas[i]:.3f} : Weight={weight:.4f}")

    print("\n--- Experiment finished ---")
