import os
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 特徴量抽出関数（MFCC）
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) == 0:
            print(f"⚠ 無音 or 読み込み失敗: {file_path}")
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc, axis=1)
    except Exception as e:
        print(f"❌ 読み込みエラー: {file_path}, {e}")
        return None

# 学習データ読み込み
X = []
y = []
vowels = ['a', 'i', 'u', 'e', 'o']
for label in vowels:
    folder = f'output_wav/{label}'
    for file in os.listdir(folder):
        if file.lower().endswith('.wav'):
            path = os.path.join(folder, file)
            features = extract_features(path)
            if features is not None:
                X.append(features)
                y.append(label)

# 配列変換（学習データが空の場合は警告）
if len(X) == 0:
    raise ValueError("❌ 学習データが空です。ファイルを確認してください。")

X = np.array(X)
y = np.array(y)

# モデル学習（スケーリング + SVM）
model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
model.fit(X, y)

# テスト音声ファイルの予測
test_folder = 'output_wav/test'
for file in os.listdir(test_folder):
    if file.lower().endswith('.wav'):
        test_path = os.path.join(test_folder, file)
        features = extract_features(test_path)
        if features is not None:
            pred = model.predict(features.reshape(1, -1))[0]
            print(f"🎤 テスト音声 {file} の予測結果: {pred}")
