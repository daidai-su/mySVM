import os
from pydub import AudioSegment

# 入力ディレクトリ（母音ごとにサブフォルダがある）
input_base = "input_m4a"
# 出力ディレクトリ
output_dir = "output_wav"
os.makedirs(output_dir, exist_ok=True)

# 母音リスト
vowels = ["a", "i", "u", "e", "o"]
target_duration_ms = 1000  # 1秒に統一

for vowel in vowels:
    folder = os.path.join(input_base, vowel)
    files = sorted([f for f in os.listdir(folder) if f.endswith(".m4a")])
    
    for idx, file in enumerate(files):
        input_path = os.path.join(folder, file)
        output_path = os.path.join(output_dir, f"{vowel}_{idx+1:02d}.wav")
        
        audio = AudioSegment.from_file(input_path, format="m4a")

        # 1秒に調整（切る or 無音追加）
        if len(audio) > target_duration_ms:
            audio = audio[:target_duration_ms]
        else:
            silence = AudioSegment.silent(duration=target_duration_ms - len(audio))
            audio = audio + silence
        
        audio.export(output_path, format="wav")
        print(f"✔️ {output_path} 書き出し完了")
