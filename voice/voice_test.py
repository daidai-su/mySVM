import os
from pydub import AudioSegment

# 入出力フォルダのパス
input_folder = "input_m4a/test"
output_folder = "output_wav/test"
os.makedirs(output_folder, exist_ok=True)

# 目標の長さ（1秒 = 1000ミリ秒）
target_duration = 1000

# ファイルごとに処理
for filename in os.listdir(input_folder):
    if filename.endswith(".m4a"):
        input_path = os.path.join(input_folder, filename)
        output_name = os.path.splitext(filename)[0] + ".wav"
        output_path = os.path.join(output_folder, output_name)

        # 音声ファイルを読み込み
        audio = AudioSegment.from_file(input_path, format="m4a")

        # 長さを1秒に調整（切り詰め or パディング）
        if len(audio) > target_duration:
            audio = audio[:target_duration]
        else:
            silence = AudioSegment.silent(duration=target_duration - len(audio))
            audio += silence

        # 書き出し（16bit PCMで保存）
        audio.export(output_path, format="wav")

print("変換完了！")
