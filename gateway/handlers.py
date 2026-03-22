import os
import json
import shutil
import sys
import tempfile
import uuid

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import QUEUE_DIR, TRANSCRIPT_DIR, SUMMARY_DIR, LOCK_FILE, PROJECT_ROOT, KB_DIR, PROCESSED_DIR, SUMMARIZATION_METHODS

from formatting import (
    format_segments, is_valid_segment, render_topics, render_word_stats,
    render_progress_bar, render_asr_progress,
)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "storage"))
from kb import load_kb_entry

sys.path.insert(0, os.path.join(PROJECT_ROOT, "rag"))
from engine import index_entry

# Сколько значений возвращает poll_outputs (синхронизировать с app.py)
_N = 8


def _no_update():
    return tuple(gr.update() for _ in range(_N))


def _write_meta(file_uuid, report_mode, sum_method="Hierarchical"):
    meta = {
        "structured": report_mode == "Структурированный отчёт",
        "sum_method": sum_method if sum_method in SUMMARIZATION_METHODS else "Hierarchical",
    }
    path = os.path.join(TRANSCRIPT_DIR, f"{file_uuid}.meta")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def save_media(file_path, report_mode="Конспект", sum_method="Hierarchical"):
    if file_path is None:
        return "Файл не выбран", None, "", gr.update(), ""

    file_uuid = str(uuid.uuid4())
    ext = os.path.splitext(file_path)[1]
    dest = os.path.join(QUEUE_DIR, f"{file_uuid}{ext}")
    shutil.copy(file_path, dest)
    _write_meta(file_uuid, report_mode, sum_method)
    return "Файл поставлен в очередь...", file_uuid, "", gr.update(), ""


def save_text(text_input, text_file, report_mode="Конспект", sum_method="Hierarchical"):
    text_content = None
    if text_file is not None:
        with open(text_file, "r", encoding="utf-8") as f:
            text_content = f.read().strip()
    elif text_input:
        text_content = text_input.strip()

    if not text_content:
        return "Текст не введён", None, "", gr.update(), ""

    file_uuid = str(uuid.uuid4())
    segments = [{"transcription": text_content, "boundaries": [0, 0]}]
    path = os.path.join(TRANSCRIPT_DIR, f"{file_uuid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False)
    _write_meta(file_uuid, report_mode, sum_method)
    return "Текст отправлен на обработку", file_uuid, "", gr.update(), ""


def save_to_kb(file_uuid, segments, summary_text, title=None, topics=None, structured_report=""):
    from datetime import datetime

    if not title:
        plain = " ".join(
            seg["transcription"] for seg in segments
            if is_valid_segment(seg["transcription"])
        )
        title = plain[:60].strip()
        if len(plain) > 60:
            title += "..."

    entry = {
        "id": file_uuid,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "title": title,
        "segments": segments,
        "summary": summary_text,
        "topics": topics or [],
        "structured_report": structured_report,
    }
    path = os.path.join(KB_DIR, f"{file_uuid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

    # добавить в RAG-индекс
    index_entry(entry)


def make_export_file(entry_id):
    """Генерирует Markdown-файл конспекта и возвращает путь к нему."""
    entry = load_kb_entry(entry_id)
    if not entry:
        return None

    title = entry.get("title", "Без названия")
    created = entry.get("created_at", "")[:16].replace("T", " ")
    topics = entry.get("topics", [])
    summary = entry.get("summary", "")
    segments = entry.get("segments", [])

    lines = [f"# {title}", "", f"**Дата:** {created}"]
    if topics:
        lines.append(f"**Темы:** {', '.join(topics)}")
    lines += ["", "---", "", "## Конспект", "", summary, "", "---", "", "## Транскрипция", ""]

    has_time = any(s["boundaries"] != [0, 0] for s in segments)
    for seg in segments:
        text = seg.get("transcription", "")
        if not is_valid_segment(text):
            continue
        if has_time:
            b = seg["boundaries"]
            t = int(b[0])
            lines.append(f"**{t // 60:02d}:{t % 60:02d}** — {text}  ")
        else:
            lines.append(text + "  ")

    md_content = "\n".join(lines)

    tmp_dir = tempfile.mkdtemp()
    safe_title = "".join(c for c in title if c.isalnum() or c in " -_")[:40].strip() or "lecture"
    fname = os.path.join(tmp_dir, f"{safe_title}.md")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(md_content)
    return fname


def poll_outputs(file_uuid, mode):
    """
    Возвращает ровно _N значений:
      0 transcript_text  → t["output"]
      1 summary_text     → t["summary_output"]
      2 uuid / None      → state["current_uuid"]
      3 segments         → state["cached_segments"]
      4 topics_html      → t["topics_html"]
      5 stats_html       → t["stats_html"]
      6 export_file      → t["export_file"]
    """
    if not file_uuid:
        return _no_update()

    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{file_uuid}.json")
    summary_path = os.path.join(SUMMARY_DIR, f"{file_uuid}.json")

    if os.path.exists(transcript_path) and os.path.exists(summary_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        shutil.move(transcript_path, os.path.join(PROCESSED_DIR, f"{file_uuid}_transcript.json"))
        shutil.move(summary_path, os.path.join(PROCESSED_DIR, f"{file_uuid}_summary.json"))

        summary_text = summary_data.get("summary", "")
        structured_report = summary_data.get("structured_report", "")
        title = summary_data.get("title")
        topics = summary_data.get("topics", [])
        save_to_kb(file_uuid, segments, summary_text, title, topics, structured_report)

        transcript_text = format_segments(segments, mode)
        topics_html = render_topics(topics)
        stats_html = render_word_stats(segments)
        export_path = make_export_file(file_uuid)
        export_update = gr.update(value=export_path, visible=bool(export_path))

        return transcript_text, summary_text, file_uuid, segments, topics_html, stats_html, export_update, structured_report

    progress_path = os.path.join(SUMMARY_DIR, f"{file_uuid}.progress")
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                prog = json.load(f)
            stage = prog.get("stage", "")
            cur = prog.get("current", 0)
            total = prog.get("total", 0)
            progress_html = render_progress_bar(stage, cur, total)
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), progress_html, gr.update(), gr.update()
        except Exception:
            pass

    if os.path.exists(LOCK_FILE):
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), render_asr_progress(), gr.update(), gr.update()

    if os.path.exists(transcript_path) and not os.path.exists(summary_path):
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), render_asr_progress(), gr.update(), gr.update()

    return _no_update()


def reformat(segments, mode):
    if not segments:
        return gr.update()
    return format_segments(segments, mode)
