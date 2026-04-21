"""
ASR-воркер. Смотрит папку queue/, транскрибирует аудиофайлы через GigaAM, кладёт результат в transcript/ как UUID.json с сегментами.

GPU-координация через gpu_coord: уступает GPU LLM-воркеру если тот просит. Пишет .asr.progress-файл чтобы UI мог показывать прогресс-бар.
"""

import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import (
    ASR_IDLE_TIMEOUT_SEC,
    ASR_MODEL,
    FFMPEG_PATH,
    LLM_MODEL,
    LOCK_FILE,
    OLLAMA_URL,
    QUEUE_DIR,
    SUMMARY_DIR,
    TRANSCRIPT_DIR,
)
from shared.gpu_coord import acquire_gpu, clear_gpu_request, read_gpu_state, release_gpu, request_gpu
from shared.log import write_event

if sys.platform == 'win32':
    os.environ["PATH"] += os.path.pathsep + FFMPEG_PATH
    os.add_dll_directory(FFMPEG_PATH)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}


def _asr_progress_path(file_uuid: str) -> str:
    return os.path.join(SUMMARY_DIR, f"{file_uuid}.asr.progress")


def _write_asr_progress(file_uuid: str, current: int, total: int) -> None:
    with open(_asr_progress_path(file_uuid), "w", encoding="utf-8") as f:
        json.dump({"stage": "asr", "current": current, "total": total}, f)


def _remove_asr_progress(file_uuid: str) -> None:
    path = _asr_progress_path(file_uuid)
    if os.path.exists(path):
        os.remove(path)


def _unload_ollama():
    """Выгружает LLM из VRAM перед загрузкой ASR-модели."""
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


def transcribe_file(model, audio_path: str, file_uuid: str | None = None) -> list:
    from gigaam.preprocess import SAMPLE_RATE
    from gigaam.vad_utils import segment_audio_file

    segments, boundaries = segment_audio_file(audio_path, SAMPLE_RATE, device=model._device)
    total = max(1, len(boundaries))
    if file_uuid:
        _write_asr_progress(file_uuid, 0, total)

    transcribed_segments = []
    for idx, (segment, segment_boundaries) in enumerate(zip(segments, boundaries), start=1):
        wav = segment.to(model._device).unsqueeze(0).to(model._dtype)
        length = torch.full([1], wav.shape[-1], device=model._device)
        encoded, encoded_len = model.forward(wav, length)
        result = model.decoding.decode(model.head, encoded, encoded_len)[0]
        if result.replace("⁇", "").replace(" ", "").strip():
            transcribed_segments.append(
                {"transcription": result, "boundaries": segment_boundaries}
            )
        if file_uuid:
            _write_asr_progress(file_uuid, idx, total)

    return transcribed_segments


def unload_model(model):
    """Выгружает модель из GPU-памяти максимально полно."""
    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _release_asr_resources(model):
    write_event("ASR", "Очередь пуста, выгружаю модель...")
    unload_model(model)
    time.sleep(1)
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
    release_gpu("asr")
    write_event("ASR", "Модель выгружена, GPU свободен")


def run_worker():
    write_event("ASR", "Воркер запущен, ожидание файлов...")
    model = None
    last_activity = 0.0

    while True:
        audio_files = [
            f for f in glob.glob(os.path.join(QUEUE_DIR, "*"))
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
        ]

        if not audio_files:
            clear_gpu_request("asr")
            if model is not None:
                state = read_gpu_state()
                llm_waiting = (state.get("requests") or {}).get("llm")
                idle_expired = (time.time() - last_activity) >= ASR_IDLE_TIMEOUT_SEC
                if llm_waiting or idle_expired:
                    _release_asr_resources(model)
                    model = None
            time.sleep(2)
            continue

        request_gpu("asr")
        if model is None:
            state = read_gpu_state()
            if state.get("owner") not in (None, "asr"):
                time.sleep(1)
                continue
            if not acquire_gpu("asr"):
                time.sleep(1)
                continue

            write_event("ASR", "Загружаю модель GigaAM...")
            open(LOCK_FILE, "w").close()
            _unload_ollama()
            model = load_model()
            last_activity = time.time()
            write_event("ASR", "Модель загружена")

        for audio_path in audio_files:
            fname = os.path.basename(audio_path)
            file_uuid = os.path.splitext(fname)[0]
            write_event("ASR", f"Транскрибирую: {fname}")
            t_start = time.time()
            try:
                segments = transcribe_file(model, audio_path, file_uuid=file_uuid)
                result = json.dumps(segments, ensure_ascii=False)
            except Exception as e:
                write_event("ASR", f"Ошибка транскрипции: {e}")
                result = json.dumps(
                    [{"transcription": f"Ошибка транскрипции: {e}", "boundaries": [0, 0]}],
                    ensure_ascii=False,
                )
            finally:
                _remove_asr_progress(file_uuid)

            elapsed = time.time() - t_start
            result_path = os.path.join(TRANSCRIPT_DIR, f"{file_uuid}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(result)
            os.remove(audio_path)
            seg_count = len(json.loads(result))
            last_activity = time.time()
            write_event("ASR", f"Готово: {seg_count} сегментов за {elapsed:.1f}с")

            if (read_gpu_state().get("requests") or {}).get("llm"):
                break

        time.sleep(0.1)


if __name__ == "__main__":
    try:
        run_worker()
    except KeyboardInterrupt:
        pass
    finally:
        clear_gpu_request("asr")
        release_gpu("asr")
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
        write_event("ASR", "Воркер остановлен")
