import io
import json
import os
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

# Загружаем без автодекодирования аудио — datasets не будет дёргать torchcodec
ds = load_dataset(
    "bond005/sberdevices_golos_10h_crowd",
    split="test",
    streaming=True,
    trust_remote_code=True,
)
ds = ds.cast_column("audio", Audio(decode=False))

os.makedirs("data/benchmark/asr/audio", exist_ok=True)

manifest = []
for i, s in enumerate(ds):
    if i >= 50:
        break

    audio_bytes = s["audio"]["bytes"]
    audio_array, sr = sf.read(io.BytesIO(audio_bytes))
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)  # стерео → моно

    path = f"data/benchmark/asr/audio/{i:04d}.wav"
    sf.write(path, audio_array.astype(np.float32), sr)
    manifest.append({"id": i, "audio": path, "transcription": s["transcription"]})
    print(f"  [{i + 1}/50] {path}")

with open("data/benchmark/asr/samples.jsonl", "w", encoding="utf-8") as f:
    for m in manifest:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

print(f"\nГотово: {len(manifest)} файлов → data/benchmark/asr/samples.jsonl")
