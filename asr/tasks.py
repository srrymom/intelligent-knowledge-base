import os
import sys
import time
import glob
import gigaam

if sys.platform == 'win32':
    ffmpeg_path = r"D:\ffmpeg\bin" 

    os.environ["PATH"] += os.path.pathsep + ffmpeg_path
    os.add_dll_directory(ffmpeg_path)

import torchcodec



UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}

print("ASR worker started, loading model...")
model = gigaam.load_model("v3_e2e_rnnt")
print("Model loaded, polling uploads/...")

while True:
    audio_files = [
        f for f in glob.glob(os.path.join(UPLOADS_DIR, "*"))
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    ]

    for audio_path in audio_files:
        file_uuid = os.path.splitext(os.path.basename(audio_path))[0]
        print(f"Transcribing: {audio_path}")
        try:
            import json
            segments = model.transcribe_longform(audio_path)
            result = json.dumps(segments, ensure_ascii=False)
        except Exception as e:
            import json
            result = json.dumps([{"transcription": f"Ошибка транскрипции: {e}", "boundaries": [0, 0]}], ensure_ascii=False)
        print(result)
        result_path = os.path.join(OUTPUTS_DIR, f"{file_uuid}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(result)

        os.remove(audio_path)
        print(f"Done: {file_uuid}.txt")

    time.sleep(2)
