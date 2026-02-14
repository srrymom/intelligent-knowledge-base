import os
import sys
import time
import glob
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import QUEUE_DIR, TRANSCRIPT_DIR, LOCK_FILE, FFMPEG_PATH, ASR_MODEL

if sys.platform == 'win32':
    os.environ["PATH"] += os.path.pathsep + FFMPEG_PATH
    os.add_dll_directory(FFMPEG_PATH)

import torchcodec  # noqa: E402
import gigaam  # noqa: E402

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}

print("ASR worker started, polling queue/...")

model = None

while True:
    audio_files = [
        f for f in glob.glob(os.path.join(QUEUE_DIR, "*"))
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    ]

    if not audio_files:
        time.sleep(2)
        continue

    if model is None:
        print("Loading ASR model...")
        open(LOCK_FILE, "w").close()
        model = gigaam.load_model(ASR_MODEL)
        print("Model loaded.")

    for audio_path in audio_files:
        file_uuid = os.path.splitext(os.path.basename(audio_path))[0]
        print(f"Transcribing: {audio_path}")
        t_start = time.time()
        try:
            segments = model.transcribe_longform(audio_path)
            segments = [
                s for s in segments
                if s["transcription"].replace("⁇", "").replace(" ", "").strip()
            ]
            result = json.dumps(segments, ensure_ascii=False)
        except Exception as e:
            result = json.dumps([{"transcription": f"Ошибка транскрипции: {e}", "boundaries": [0, 0]}], ensure_ascii=False)
        elapsed = time.time() - t_start
        print(result)
        result_path = os.path.join(TRANSCRIPT_DIR, f"{file_uuid}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(result)

        os.remove(audio_path)
        print(f"Done: {file_uuid}.json ({elapsed:.1f}s)")

    print("Queue empty, unloading ASR model...")
    del model
    model = None
    torch.cuda.empty_cache()
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
    print("ASR model unloaded, GPU/RAM free.")

    time.sleep(2)
