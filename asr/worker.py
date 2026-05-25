"""
ASR-воркер. Смотрит папку queue/, транскрибирует аудиофайлы через GigaAM, кладёт результат в transcript/ как UUID.json с сегментами.

GPU-координация через gpu_coord: уступает GPU LLM-воркеру если тот просит. Пишет .asr.progress-файл чтобы UI мог показывать прогресс-бар.
"""

import glob
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import (
    ASR_IDLE_TIMEOUT_SEC,
    ASR_MODEL,
    ASR_WORKER_PID_FILE,
    FFMPEG_PATH,
    LLM_MODEL,
    LOCK_FILE,
    OLLAMA_URL,
    QUEUE_DIR,
    QUEUE_PROCESSING_DIR,
    SUMMARY_DIR,
    TRANSCRIPT_DIR,
)
from shared.gpu_coord import acquire_gpu, clear_gpu_request, read_gpu_state, release_gpu, request_gpu
from shared.log import write_event, write_resource_event
from shared.process_singleton import singleton_process

if sys.platform == 'win32':
    os.environ["PATH"] += os.path.pathsep + FFMPEG_PATH
    os.add_dll_directory(FFMPEG_PATH)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
ASR_MAX_SUBCHUNK_SEC = 15.0
ASR_COVERAGE_WARN_GAP_PCT = 10.0
UNKNOWN_ASR_CHARS = {"⁇", "�"}


def _process_diag() -> str:
    return (
        f"pid={os.getpid()} ppid={os.getppid()} "
        f"exe={sys.executable} argv={' '.join(sys.argv)}"
    )


def _claim_audio_file(audio_path: str) -> str | None:
    fname = os.path.basename(audio_path)
    stem, ext = os.path.splitext(fname)
    claimed_path = os.path.join(QUEUE_PROCESSING_DIR, f"{stem}.{os.getpid()}{ext}")
    try:
        os.replace(audio_path, claimed_path)
    except FileNotFoundError:
        return None
    except OSError as e:
        write_event("ASR", f"DIAG CLAIM_FAILED file={fname} error={e} {_process_diag()}")
        return None
    write_event("ASR", f"DIAG CLAIM_OK file={fname} claimed_path={claimed_path} {_process_diag()}")
    return claimed_path


def _claimed_processing_files() -> list[tuple[str, str]]:
    claimed = []
    pattern = os.path.join(QUEUE_PROCESSING_DIR, "*")
    for claimed_path in glob.glob(pattern):
        if os.path.splitext(claimed_path)[1].lower() not in AUDIO_EXTENSIONS:
            continue
        claimed_name = os.path.basename(claimed_path)
        stem, ext = os.path.splitext(claimed_name)
        original_stem = stem.rsplit(".", 1)[0]
        claimed.append((f"{original_stem}{ext}", claimed_path))
    return claimed


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


def _format_asr_time(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    minutes, rem_ms = divmod(total_ms, 60_000)
    sec, ms = divmod(rem_ms, 1000)
    return f"{minutes:02d}:{sec:02d}.{ms:03d}"


def _clean_asr_text(text: str) -> str:
    cleaned = text or ""
    for ch in UNKNOWN_ASR_CHARS:
        cleaned = cleaned.replace(ch, "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _assess_asr_text(text: str, duration: float) -> tuple[bool, str, int, str]:
    unknown_count = sum((text or "").count(ch) for ch in UNKNOWN_ASR_CHARS)
    cleaned = _clean_asr_text(text)
    compact = re.sub(r"\s+", "", cleaned)
    word_count = len(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", cleaned))
    lowered = cleaned.lower()

    if "очень длинная строка" in lowered and unknown_count >= 3:
        return False, "gigaam_long_unknown_string", unknown_count, cleaned
    if not compact:
        if unknown_count > 0:
            return False, "only_unknown_tokens", unknown_count, cleaned
        return False, "empty_text", unknown_count, cleaned
    if unknown_count >= 5 and len(compact) < 40:
        return False, "many_unknowns_low_text", unknown_count, cleaned
    if duration >= 8.0 and word_count <= 2:
        return False, "too_few_words_for_long_audio", unknown_count, cleaned
    if unknown_count >= 3:
        unknown_ratio = unknown_count / max(1, unknown_count + len(compact))
        if unknown_ratio >= 0.20:
            return False, f"high_unknown_ratio_{unknown_ratio:.2f}", unknown_count, cleaned
    return True, "ok", unknown_count, cleaned


def _split_grouped_segment(
    segment,
    boundaries: tuple[float, float],
    sample_rate: int,
) -> list[tuple[object, tuple[float, float]]]:
    start, end = float(boundaries[0]), float(boundaries[1])
    duration = max(0.0, end - start)
    if duration <= ASR_MAX_SUBCHUNK_SEC:
        return [(segment, (start, end))]

    parts = max(1, math.ceil(duration / ASR_MAX_SUBCHUNK_SEC))
    samples = int(segment.shape[-1])
    subchunks = []
    for part_idx in range(parts):
        sub_start = start + duration * part_idx / parts
        sub_end = start + duration * (part_idx + 1) / parts
        sample_start = int(round((sub_start - start) * sample_rate))
        sample_end = (
            samples
            if part_idx == parts - 1
            else int(round((sub_end - start) * sample_rate))
        )
        chunk = segment[sample_start:sample_end]
        if int(chunk.shape[-1]) <= 0:
            continue
        subchunks.append((chunk, (sub_start, sub_end)))
    return subchunks


def _coverage_duration(segments: list[dict], *, valid_only: bool | None = None) -> float:
    intervals = []
    for seg in segments:
        if valid_only is not None and bool(seg.get("asr_valid", True)) != valid_only:
            continue
        try:
            start, end = seg["boundaries"]
        except (KeyError, TypeError, ValueError):
            continue
        start = max(0.0, float(start))
        end = max(0.0, float(end))
        if end > start:
            intervals.append((start, end))

    intervals.sort()
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 0.001:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def transcribe_file(model, audio_path: str, file_uuid: str | None = None) -> list:
    import torch
    from gigaam.preprocess import SAMPLE_RATE, load_audio
    from gigaam.vad_utils import segment_audio_file

    total_audio_duration = int(load_audio(audio_path).shape[-1]) / SAMPLE_RATE
    segments, boundaries = segment_audio_file(audio_path, SAMPLE_RATE, device=model._device)
    decode_items = []
    for segment, segment_boundaries in zip(segments, boundaries):
        decode_items.extend(_split_grouped_segment(segment, segment_boundaries, SAMPLE_RATE))

    total = max(1, len(decode_items))
    if file_uuid:
        _write_asr_progress(file_uuid, 0, total)
    write_event(
        "ASR",
        f"DIAG ASR_SEGMENTATION grouped={len(boundaries)} subchunks={len(decode_items)} "
        f"max_subchunk={ASR_MAX_SUBCHUNK_SEC:.1f}s audio={total_audio_duration:.3f}s",
    )

    transcribed_segments = []
    for idx, (segment, segment_boundaries) in enumerate(decode_items, start=1):
        start, end = segment_boundaries
        duration = max(0.0, float(end) - float(start))
        wav = segment.to(model._device).unsqueeze(0).to(model._dtype)
        length = torch.full([1], wav.shape[-1], device=model._device)
        encoded, encoded_len = model.forward(wav, length)
        result = model.decoding.decode(model.head, encoded, encoded_len)[0]
        is_valid, invalid_reason, unknown_count, cleaned = _assess_asr_text(result, duration)
        if not is_valid:
            write_event(
                "ASR",
                "WARNING invalid ASR segment "
                f"{_format_asr_time(start)}-{_format_asr_time(end)} "
                f"duration={duration:.3f}s reason={invalid_reason} unknown={unknown_count}",
            )
        transcribed_segments.append(
            {
                "transcription": cleaned if is_valid else "",
                "boundaries": [float(start), float(end)],
                "asr_valid": is_valid,
                "asr_invalid_reason": "" if is_valid else invalid_reason,
                "asr_raw_transcription": "" if is_valid else result,
            }
        )
        if file_uuid:
            _write_asr_progress(file_uuid, idx, total)

    audio_coverage_sec = _coverage_duration(transcribed_segments)
    valid_text_duration = _coverage_duration(transcribed_segments, valid_only=True)
    invalid_duration = _coverage_duration(transcribed_segments, valid_only=False)
    total_duration = max(total_audio_duration, 0.001)
    audio_coverage_pct = audio_coverage_sec / total_duration * 100.0
    valid_coverage_pct = valid_text_duration / total_duration * 100.0
    invalid_coverage_pct = invalid_duration / total_duration * 100.0
    coverage_msg = (
        f"ASR coverage: audio={audio_coverage_pct:.1f}% "
        f"valid={valid_coverage_pct:.1f}% invalid={invalid_coverage_pct:.1f}% "
        f"valid_sec={valid_text_duration:.3f}/{total_audio_duration:.3f}"
    )
    write_event("ASR", coverage_msg)
    if audio_coverage_pct - valid_coverage_pct >= ASR_COVERAGE_WARN_GAP_PCT:
        write_event(
            "ASR",
            "WARNING valid ASR coverage заметно ниже audio coverage: "
            f"audio={audio_coverage_pct:.1f}% valid={valid_coverage_pct:.1f}% "
            f"gap={audio_coverage_pct - valid_coverage_pct:.1f}%",
        )

    return transcribed_segments


def unload_model(model):
    """Выгружает модель из GPU-памяти максимально полно."""
    import gc
    import torch
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
    write_resource_event("ASR", "Перед выгрузкой GigaAM")
    unload_model(model)
    time.sleep(1)
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
    release_gpu("asr")
    write_event("ASR", "Модель выгружена, GPU свободен")
    write_resource_event("ASR", "После выгрузки GigaAM")


def run_worker():
    write_event("ASR", f"DIAG worker_start {_process_diag()}")
    write_event("ASR", "Воркер запущен, ожидание файлов...")
    write_resource_event("ASR", "Старт ASR-воркера")
    model = None
    last_activity = 0.0

    while True:
        audio_files = [
            f for f in glob.glob(os.path.join(QUEUE_DIR, "*"))
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
        ]
        claimed_files = _claimed_processing_files()
        for audio_path in audio_files:
            claimed_path = _claim_audio_file(audio_path)
            if claimed_path:
                claimed_files.append((os.path.basename(audio_path), claimed_path))

        if not claimed_files:
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
            write_resource_event("ASR", "Перед загрузкой GigaAM")
            open(LOCK_FILE, "w").close()
            _unload_ollama()
            model = load_model()
            last_activity = time.time()
            write_event("ASR", "Модель загружена")
            write_resource_event("ASR", "После загрузки GigaAM")

        for fname, audio_path in claimed_files:
            file_uuid = os.path.splitext(fname)[0]
            write_event("ASR", f"DIAG WORKER_CLAIM file={fname} uuid={file_uuid} {_process_diag()}")
            write_event("ASR", f"Транскрибирую: {fname}")
            write_resource_event("ASR", f"Перед транскрипцией {fname}")
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
            write_event("ASR", f"DIAG WRITE_TRANSCRIPT uuid={file_uuid} path={result_path} {_process_diag()}")
            os.remove(audio_path)
            write_event("ASR", f"DIAG REMOVE_CLAIMED_FILE uuid={file_uuid} path={audio_path} {_process_diag()}")
            seg_count = len(json.loads(result))
            last_activity = time.time()
            write_event("ASR", f"Готово: {seg_count} сегментов за {elapsed:.1f}с")
            write_resource_event("ASR", f"После транскрипции {fname}")

            if (read_gpu_state().get("requests") or {}).get("llm"):
                break

        time.sleep(0.1)


if __name__ == "__main__":
    with singleton_process("asr_worker", ASR_WORKER_PID_FILE, "ASR") as acquired:
        if not acquired:
            sys.exit(0)
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
