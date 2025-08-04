import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics.pairwise import rbf_kernel

# --- データセット生成関数群 -------------------------------------------------
def generate_blobs(n_samples=300, centers=2, random_state=42):
    return make_blobs(n_samples=n_samples, centers=centers, random_state=random_state, cluster_std=1.2)

def generate_moons(n_samples=300, noise=0.5, random_state=42):
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

def generate_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42):
    return make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)

def generate_cross(n_samples=400, noise=0.15, random_state=42):
    centers = [[-2, 2], [2, -2], [-2, -2], [2, 2]]
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=noise*3, random_state=random_state)
    y = y % 2
    angle = np.pi / 4
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return X.dot(rotation_matrix), y

def generate_checkerboard(n_samples=900, n_grid=3, noise=0.3, random_state=42):
    n_clusters = n_grid * n_grid
    grid_coords = np.linspace(-n_grid, n_grid, n_grid)
    centers = [[x, y] for x in grid_coords for y in grid_coords]
    X, y_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=noise*3, random_state=random_state)
    y = (y_true // n_grid + y_true % n_grid) % 2
    return X, y

# ★★★ 新しく追加した関数 ★★★
def generate_stripes(n_samples=400, n_stripes=10, noise=0.15, random_state=42):
    """
    クラスが交互に並ぶ、縞模様のデータセットを生成します。
    """
    np.random.seed(random_state)
    X_ideal = np.random.rand(n_samples, 2)
    X_ideal[:, 0] *= n_stripes
    X_ideal[:, 1] *= 5
    y = np.floor(X_ideal[:, 0]).astype(int) % 2
    X = X_ideal + np.random.randn(n_samples, 2) * noise
    return X, y

def generate_sigmoid_shape(n_samples=400, noise=0.15, random_state=42):
    """
    決定境界がS字状になるデータセットを生成します。
    シグモイドカーネルの性能をテストするのに適しています。
    """
    np.random.seed(random_state)
    X = np.random.rand(n_samples, 2) * 10 - 5
    y = np.zeros(n_samples, dtype=int)
    
    # tanh(x)を境界線とする
    boundary = np.tanh(X[:, 0]) * 3
    y[X[:, 1] > boundary] = 1
    
    X += np.random.randn(n_samples, 2) * noise
    return X, y

def generate_concentric_squares(n_samples=500, noise=0.1, random_state=42):
    """
    同心円ならぬ「同心正方形」のデータセットを生成します。
    RBFカーネルが苦手な、角のある境界の学習能力をテストします。
    """
    np.random.seed(random_state)
    X = np.random.rand(n_samples, 2) * 6 - 3
    y = np.ones(n_samples, dtype=int)
    
    # 内側の正方形（クラス0）
    inner_mask = (np.abs(X[:, 0]) < 1) & (np.abs(X[:, 1]) < 1)
    y[inner_mask] = 0
    
    # 外側の正方形の外側は捨てる
    outer_mask = (np.abs(X[:, 0]) < 2.5) & (np.abs(X[:, 1]) < 2.5)
    X = X[outer_mask]
    y = y[outer_mask]
    
    X += np.random.randn(X.shape[0], 2) * noise
    return X, y

def generate_satellite_clusters(n_samples=500, noise=0.2, random_state=42):
    """
    中央の大きなクラスターと、それを取り囲む衛星のような
    小さなクラスター群を分離させるデータセット。
    """
    np.random.seed(random_state)
    # 中央の大きな塊 (クラス0)
    X_center, y_center = make_blobs(n_samples=n_samples//2, centers=[[0, 0]], 
                                    cluster_std=1.0, random_state=random_state)
    
    # 周囲の小さな塊 (クラス1)
    n_satellites = 5
    satellite_samples = (n_samples // 2) // n_satellites
    X_sat = []
    for i in range(n_satellites):
        angle = 2 * np.pi * i / n_satellites
        center = [4 * np.cos(angle), 4 * np.sin(angle)]
        X_s, _ = make_blobs(n_samples=satellite_samples, centers=[center],
                            cluster_std=noise*3, random_state=random_state + i)
        X_sat.append(X_s)
        
    X_sat = np.vstack(X_sat)
    y_sat = np.ones(X_sat.shape[0], dtype=int)
    
    X = np.vstack([X_center, X_sat])
    y = np.hstack([y_center, y_sat])
    
    return X, y

def generate_pincer(n_samples=400, noise=0.1, random_state=42):
    """
    一方のクラスがもう一方を挟み込むような、ピンサー（やっとこ）形状。
    """
    np.random.seed(random_state)
    # 中央の塊 (クラス0)
    X0 = np.random.rand(n_samples // 2, 2)
    X0[:, 0] = X0[:, 0] * 2 - 1
    X0[:, 1] = X0[:, 1] * 2 - 1
    y0 = np.zeros(n_samples // 2, dtype=int)
    
    # ピンサー部分 (クラス1)
    n_pincer = n_samples // 2
    angle = np.linspace(0.5, 2.5, n_pincer) * np.pi
    radius = 2
    x_p = radius * np.cos(angle) + np.random.randn(n_pincer) * noise * 2
    y_p = radius * np.sin(angle) + np.random.randn(n_pincer) * noise * 2
    X1 = np.vstack([x_p, y_p]).T
    y1 = np.ones(n_pincer, dtype=int)

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    return X, y

def generate_blob_with_holes(n_samples=1000, n_holes=6, noise=0.05, random_state=42):
    """
    大きなブロブの内部に、異なるクラスの小さな穴が点在するデータセット。
    単一スケールのカーネルでは両立が難しいタスク。
    """
    np.random.seed(random_state)
    
    # -4から4の範囲にデータ点をランダムに配置
    X = np.random.uniform(-4, 4, (n_samples, 2))
    
    # まず、全ての点を大きな円の中にあるものだけ残す（クラス0）
    dist_from_center = np.linalg.norm(X, axis=1)
    mask_main_blob = dist_from_center < 3.5
    X = X[mask_main_blob]
    y = np.zeros(X.shape[0], dtype=int)
    n_samples = X.shape[0]

    # 「穴」の中心点を円状に配置
    hole_centers = []
    radius = 2.0
    for i in range(n_holes):
        angle = 2 * np.pi * i / n_holes
        hole_centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    hole_centers = np.array(hole_centers)
    
    # 各点の「穴」の中心からの距離を測り、一定より近ければクラス1とする
    for i in range(n_samples):
        distances_to_holes = np.linalg.norm(X[i] - hole_centers, axis=1)
        if np.min(distances_to_holes) < 0.6:
            y[i] = 1
            
    # 全体にノイズを追加
    X += np.random.randn(n_samples, 2) * noise
    
    return X, y

# --- MKLカスタムカーネル ---------------------------------------------------
def multi_scale_rbf_kernel(X1, X2):
    #gammas = np.logspace(-3, 3, 30)
    gammas = np.linspace(0, 1000, 30)
    weights = np.linspace(0, 100, 30)
    kernel_sum = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(0,30):
        kernel_sum += weights[i] * rbf_kernel(X1, X2, gamma=gammas[i])
    return kernel_sum

# --- ここで実験するデータセットを選択 ---------------------------------------
# 'blobs', 'moons', 'circles', 'cross', 'checkerboard', 'stripes' から選んでください
SELECTED_DATASET = 'stripes' 
# --------------------------------------------------------------------------

# データセット生成関数の辞書
dataset_generators = {
    'blobs': generate_blobs,
    'moons': generate_moons,
    'circles': generate_circles,
    'cross': generate_cross, #しょうもない
    'checkerboard': generate_checkerboard,
    'stripes': generate_stripes,
    'sigmoid': generate_sigmoid_shape,
    'squares': generate_concentric_squares, #しょうもない
    'satellite': generate_satellite_clusters,#しょうもない
    'pincer': generate_pincer, #しょうもない
    'holes': generate_blob_with_holes
}

# 選択されたデータセットを生成
X, y = dataset_generators[SELECTED_DATASET]()

# データの準備
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# カーネルの定義
kernels = {
    'Linear':    {'kernel': 'linear', 'C': 1.0},
    'Poly':      {'kernel': 'poly', 'degree': 5, 'C': 1.0},
    'RBF':       {'kernel': 'rbf', 'gamma': 10, 'C': 1.0},
    'Sigmoid':   {'kernel': 'sigmoid', 'gamma': 'auto', 'C': 1.0},
    'MKL_RBF':   {'kernel': multi_scale_rbf_kernel, 'C': 100}
}

# 決定境界のプロット関数
def plot_decision_boundary(clf, X, y, scaler, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    mesh_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = clf.predict(mesh_scaled).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=25)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

# --- メイン処理: 学習と可視化 ---------------------------------------------
# 5つのモデルに加えて、元のデータも表示するためレイアウトを3x2にする
plt.figure(figsize=(12, 16))

# 元のデータセットをプロット
ax = plt.subplot(3, 2, 1)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=25)
ax.set_title(f"Original Dataset: '{SELECTED_DATASET.capitalize()}'")
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')


for i, (name, params) in enumerate(kernels.items()):
    ax = plt.subplot(3, 2, i + 2) # 2番目のプロットから開始
    
    clf = SVC(**params, probability=True)
    clf.fit(X_train_scaled, y_train)
    score = clf.score(X_test_scaled, y_test)
    
    title = f"Kernel: {name}\nAccuracy: {score:.2f}"
    plot_decision_boundary(clf, X, y, scaler, title)

plt.suptitle(f"SVM Classification on '{SELECTED_DATASET.capitalize()}' Dataset", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
