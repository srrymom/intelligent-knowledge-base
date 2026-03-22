import os
import sys
import time
import glob
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import QUEUE_DIR, TRANSCRIPT_DIR, LOCK_FILE, FFMPEG_PATH, ASR_MODEL, OLLAMA_URL, LLM_MODEL
from shared.log import write_event

if sys.platform == 'win32':
    os.environ["PATH"] += os.path.pathsep + FFMPEG_PATH
    os.add_dll_directory(FFMPEG_PATH)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}


def _unload_ollama():
    """Выгружает LLM из VRAM перед загрузкой ASR-модели (только urllib, без зависимостей)."""
    try:
        import urllib.request
        import json as _json
        data = _json.dumps({"model": LLM_MODEL, "keep_alive": 0}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=8)
        print("Ollama: LLM выгружена из VRAM.")
    except Exception as e:
        print(f"Ollama unload: {e} (игнорируем)")


def load_model(model_name=None):
    """Загружает модель GigaAM. Импортирует torchcodec/gigaam отложенно."""
    import torchcodec  # noqa: F401
    import gigaam
    return gigaam.load_model(model_name or ASR_MODEL)


def transcribe_file(model, audio_path: str) -> list:
    """
    Транскрибирует аудиофайл моделью GigaAM.
    Возвращает список сегментов: [{"transcription": str, "boundaries": [float, float]}, ...]
    """
    segments = model.transcribe_longform(audio_path)
    return [
        s for s in segments
        if s["transcription"].replace("⁇", "").replace(" ", "").strip()
    ]


def unload_model(model):
    """Выгружает модель из GPU-памяти максимально полно."""
    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()   # ждём завершения всех CUDA-операций
        torch.cuda.empty_cache()
        # ipc_collect нужен если другой процесс держит handle на ту же память
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def run_worker():
    write_event("ASR", "Воркер запущен, ожидание файлов...")
    model = None

    while True:
        audio_files = [
            f for f in glob.glob(os.path.join(QUEUE_DIR, "*"))
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
        ]

        if not audio_files:
            if model is not None:
                write_event("ASR", "Очередь пуста, выгружаю модель...")
                unload_model(model)
                model = None
                time.sleep(1)   # даём CUDA время полностью освободить память
                if os.path.exists(LOCK_FILE):
                    os.remove(LOCK_FILE)
                write_event("ASR", "Модель выгружена, GPU свободен")
            time.sleep(2)
            continue

        if model is None:
            write_event("ASR", "Загружаю модель GigaAM...")
            open(LOCK_FILE, "w").close()
            _unload_ollama()   # освобождаем VRAM от LLM перед загрузкой ASR
            model = load_model()
            write_event("ASR", "Модель загружена")

        for audio_path in audio_files:
            fname = os.path.basename(audio_path)
            file_uuid = os.path.splitext(fname)[0]
            write_event("ASR", f"Транскрибирую: {fname}")
            t_start = time.time()
            try:
                segments = transcribe_file(model, audio_path)
                result = json.dumps(segments, ensure_ascii=False)
            except Exception as e:
                write_event("ASR", f"Ошибка транскрипции: {e}")
                result = json.dumps(
                    [{"transcription": f"Ошибка транскрипции: {e}", "boundaries": [0, 0]}],
                    ensure_ascii=False,
                )
            elapsed = time.time() - t_start
            result_path = os.path.join(TRANSCRIPT_DIR, f"{file_uuid}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(result)
            os.remove(audio_path)
            seg_count = len(json.loads(result))
            write_event("ASR", f"Готово: {seg_count} сегментов за {elapsed:.1f}с")

        time.sleep(0.1)


if __name__ == "__main__":
    run_worker()
