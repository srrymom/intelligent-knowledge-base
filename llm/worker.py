import glob
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm.summarization import check_ollama_available, get_chunk_word_limit, make_chunks, normalize_method, summarize_transcript, unload_from_vram
from shared.config import (
    DEFAULT_SUMMARIZATION_METHOD,
    KB_DIR,
    LLM_IDLE_TIMEOUT_SEC,
    LOCK_FILE,
    QUEUE_DIR,
    SUMMARY_DIR,
    TRANSCRIPT_DIR,
)
from shared.gpu_coord import acquire_gpu, clear_gpu_request, read_gpu_state, release_gpu, request_gpu
from shared.log import write_event

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".mp4", ".mkv", ".avi", ".mov"}


def write_progress(file_uuid, stage, current=0, total=0):
    path = os.path.join(SUMMARY_DIR, f"{file_uuid}.progress")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "current": current, "total": total}, f)


def remove_progress(file_uuid):
    path = os.path.join(SUMMARY_DIR, f"{file_uuid}.progress")
    if os.path.exists(path):
        os.remove(path)


def _read_meta(file_uuid: str) -> tuple[bool, str]:
    meta_path = os.path.join(TRANSCRIPT_DIR, f"{file_uuid}.meta")
    want_structured = False
    sum_method = DEFAULT_SUMMARIZATION_METHOD

    if not os.path.exists(meta_path):
        return want_structured, sum_method

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        want_structured = meta.get("structured", False)
        sum_method = normalize_method(meta.get("sum_method", DEFAULT_SUMMARIZATION_METHOD))
        os.remove(meta_path)
    except Exception:
        pass

    return want_structured, sum_method


def _pending_audio_exists() -> bool:
    return any(
        os.path.splitext(path)[1].lower() in AUDIO_EXTENSIONS
        for path in glob.glob(os.path.join(QUEUE_DIR, "*"))
    )


def _pending_transcripts() -> list[str]:
    pending = []
    for path in glob.glob(os.path.join(TRANSCRIPT_DIR, "*.json")):
        file_uuid = os.path.splitext(os.path.basename(path))[0]
        if not os.path.exists(os.path.join(KB_DIR, f"{file_uuid}.json")):
            pending.append(file_uuid)
    pending.sort()
    return pending


def _release_llm_resources():
    unload_from_vram()
    release_gpu("llm")
    write_event("LLM", "Модель выгружена, GPU освобождена")


def _process_one(file_uuid: str):
    """Обрабатывает один транскрипт и сохраняет результат в KB."""
    json_path = os.path.join(TRANSCRIPT_DIR, f"{file_uuid}.json")
    kb_path = os.path.join(KB_DIR, f"{file_uuid}.json")

    if os.path.exists(kb_path):
        write_event("LLM", f"Уже в KB: {file_uuid}")
        return

    write_event("LLM", f"Обработка: {file_uuid}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        if not segments:
            raise ValueError("Пустой транскрипт")

        chunk_word_limit = get_chunk_word_limit()
        chunks = make_chunks(segments, chunk_word_limit)
        if not chunks:
            raise ValueError("Не удалось разбить на чанки")

        write_event("LLM", f"Чанков: {len(chunks)}")

        want_structured, sum_method = _read_meta(file_uuid)
        write_event("LLM", f"Метод: {sum_method}")

        result = summarize_transcript(
            segments,
            word_limit=chunk_word_limit,
            method=sum_method,
            want_structured_report=want_structured,
            progress_callback=lambda stage, current, total: write_progress(file_uuid, stage, current, total),
            event_callback=lambda message: write_event("LLM", message),
        )

        entry = {
            "id": file_uuid,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "title": result["title"],
            "segments": segments,
            "summary": result["summary"],
            "topics": result["topics"],
            "structured_report": result["structured_report"],
            "sum_method": result["method"],
        }

    except Exception as e:
        write_event("LLM", f"Ошибка: {e}")
        entry = {
            "id": file_uuid,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "title": "Ошибка обработки",
            "segments": [],
            "summary": f"Ошибка обработки: {e}",
            "topics": [],
            "structured_report": "",
            "sum_method": "unknown",
        }

    remove_progress(file_uuid)

    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

    try:
        os.remove(json_path)
    except Exception:
        pass

    write_event("LLM", f"Готово: '{entry['title']}' [{entry['sum_method']}]")


def run_worker():
    write_event("LLM", "Воркер запущен, ожидает транскрипты...")
    llm_active = False
    last_activity = 0.0

    while True:
        pending = _pending_transcripts()

        if not pending:
            clear_gpu_request("llm")
            if llm_active:
                state = read_gpu_state()
                asr_waiting = (state.get("requests") or {}).get("asr")
                idle_expired = (time.time() - last_activity) >= LLM_IDLE_TIMEOUT_SEC
                if asr_waiting or idle_expired:
                    _release_llm_resources()
                    llm_active = False
            time.sleep(2)
            continue

        if not check_ollama_available():
            write_event("LLM", "Ollama недоступна — ожидание 30 сек...")
            clear_gpu_request("llm")
            if llm_active:
                _release_llm_resources()
                llm_active = False
            time.sleep(30)
            continue

        request_gpu("llm")
        if not llm_active:
            state = read_gpu_state()
            if state.get("owner") not in (None, "llm"):
                time.sleep(1)
                continue
            if not acquire_gpu("llm"):
                time.sleep(1)
                continue
            llm_active = True
            last_activity = time.time()
            write_event("LLM", "GPU захвачена для суммаризации")

        for file_uuid in pending:
            _process_one(file_uuid)
            last_activity = time.time()

            state = read_gpu_state()
            if _pending_audio_exists() and (state.get("requests") or {}).get("asr"):
                break

        time.sleep(0.1)


if __name__ == "__main__":
    try:
        run_worker()
    except KeyboardInterrupt:
        pass
    finally:
        clear_gpu_request("llm")
        release_gpu("llm")
        write_event("LLM", "Воркер остановлен")
