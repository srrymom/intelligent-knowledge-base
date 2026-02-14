import os
import json
import shutil
import sys
import uuid

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import QUEUE_DIR, TRANSCRIPT_DIR, SUMMARY_DIR, LOCK_FILE, PROJECT_ROOT

from formatting import format_segments, is_valid_segment

sys.path.insert(0, os.path.join(PROJECT_ROOT, "rag"))
from engine import index_entry


def save_media(file_path):
    if file_path is None:
        return "Файл не выбран", None, gr.update(), gr.update()

    file_uuid = str(uuid.uuid4())
    ext = os.path.splitext(file_path)[1]
    dest = os.path.join(QUEUE_DIR, f"{file_uuid}{ext}")
    shutil.copy(file_path, dest)
    return "Файл сохранён", file_uuid, gr.update(), gr.update()


def save_text(text_input, text_file):
    text_content = None
    if text_file is not None:
        with open(text_file, "r", encoding="utf-8") as f:
            text_content = f.read().strip()
    elif text_input:
        text_content = text_input.strip()

    if not text_content:
        return "Текст не введён", None, gr.update(), gr.update()

    file_uuid = str(uuid.uuid4())
    segments = [{"transcription": text_content, "boundaries": [0, 0]}]
    path = os.path.join(TRANSCRIPT_DIR, f"{file_uuid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False)
    return "Текст отправлен на обработку", file_uuid, gr.update(), gr.update()


def save_to_kb(file_uuid, segments, summary_text, title=None):
    from datetime import datetime

    if not title:
        plain = " ".join(
            seg["transcription"] for seg in segments
            if is_valid_segment(seg["transcription"])
        )
        title = plain[:60].strip()
        if len(plain) > 60:
            title += "..."

    from shared.config import KB_DIR

    entry = {
        "id": file_uuid,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "title": title,
        "segments": segments,
        "summary": summary_text,
    }
    path = os.path.join(KB_DIR, f"{file_uuid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

    # добавить в RAG-индекс
    index_entry(entry)


def poll_outputs(file_uuid, mode):
    if not file_uuid:
        return gr.update(), gr.update(), gr.update(), gr.update()

    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{file_uuid}.json")
    summary_path = os.path.join(SUMMARY_DIR, f"{file_uuid}.json")

    if os.path.exists(transcript_path) and os.path.exists(summary_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        os.remove(transcript_path)
        os.remove(summary_path)

        summary_text = summary_data.get("summary", "")
        title = summary_data.get("title")
        save_to_kb(file_uuid, segments, summary_text, title)

        transcript_text = format_segments(segments, mode)
        return transcript_text, summary_text, None, segments

    progress_path = os.path.join(SUMMARY_DIR, f"{file_uuid}.progress")
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                prog = json.load(f)
            stage = prog.get("stage", "")
            cur = prog.get("current", 0)
            total = prog.get("total", 0)
            pct = round(cur / total * 100) if total else 0
            if stage == "chunk":
                status = f"Суммаризация: чанк {cur}/{total - 2} ({pct}%)"
            elif stage == "merge":
                status = f"Объединение резюме... ({pct}%)"
            elif stage == "title":
                status = f"Генерация названия... ({pct}%)"
            else:
                status = f"Обработка... ({pct}%)"
            return gr.update(), status, gr.update(), gr.update()
        except Exception:
            pass

    if os.path.exists(LOCK_FILE):
        return gr.update(), "Идёт транскрипция...", gr.update(), gr.update()

    if os.path.exists(transcript_path) and not os.path.exists(summary_path):
        return gr.update(), "Ожидание суммаризации...", gr.update(), gr.update()

    return gr.update(), gr.update(), gr.update(), gr.update()


def reformat(segments, mode):
    if not segments:
        return gr.update()
    return format_segments(segments, mode)
