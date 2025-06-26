ロードマップ
🚩【STEP 1】標準データセット × SVM 4種カーネル
✅ やること
・scikit-learn のデータセット（例：Iris, make_moons, make_circles, digits など）を読み込む
・SVC(kernel=…) で以下の4種類を使って学習・可視化
　　linear
　　poly
　　rbf
　　sigmoid
・分類結果を2次元プロットで比較（decision_function or predict）

🚩【STEP 2】Multiple Kernel（複数カーネルの合成）
✅ やること
・自作カーネル関数を SVC(kernel=callable) で適用
・複数の既存カーネル（例：RBF + polynomial）を 重み付き和 or 平均で合成
・実験＆可視化（ステップ1と比較）

🚩【STEP 3】独自データセットへの応用
✅ やること
・CSVやExcel等から独自データを読み込み（pandas使用）
・前処理（欠損値除去・スケーリングなど）
・STEP 1 or STEP 2のモデルを再利用して適用・評価
・学習曲線、confusion matrix、accuracy などを確認
