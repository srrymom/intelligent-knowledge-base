"""RAG chat service functions."""

import os
import sys

from gradio import ChatMessage

from shared.config import PROJECT_ROOT

sys.path.insert(0, os.path.join(PROJECT_ROOT, "storage"))
from kb import get_kb_stats, load_kb_entry

from formatting import format_segments


def _load_rag_engine():
    rag_dir = os.path.join(PROJECT_ROOT, "rag")
    if rag_dir not in sys.path:
        sys.path.insert(0, rag_dir)
    from engine import ask as rag_ask, ensure_indexed
    return rag_ask, ensure_indexed

def chat_respond(message, history):
    if not (message or "").strip():
        return [], [], []

    kb_count, _, _ = get_kb_stats()
    if kb_count == 0:
        return (
            [ChatMessage(content="База знаний пока пуста. Сначала обработайте или добавьте лекцию.")],
            [],
            [],
        )

    rag_ask, ensure_indexed = _load_rag_engine()
    try:
        ensure_indexed()
        result = rag_ask(message)
    except Exception as e:
        return (
            [ChatMessage(content=f"Не удалось выполнить RAG-запрос: {e}")],
            [],
            [],
        )
    answer = result["answer"]
    sources = result["sources"]

    new_messages = [ChatMessage(content=answer)]
    if sources:
        sources_text = "\n".join(f"- {s['title']}" for s in sources)
        new_messages.append(
            ChatMessage(
                content=sources_text,
                metadata={"title": "Источники", "status": "done"},
            )
        )
    choices = [s["title"] for s in sources]
    ids = [s["id"] for s in sources]
    return new_messages, choices, ids


def on_source_select(title, source_ids, source_titles):
    if not title or not source_titles:
        return "", "", ""
    try:
        idx = source_titles.index(title)
    except ValueError:
        return "", "", ""
    entry = load_kb_entry(source_ids[idx])
    if entry is None:
        return "", "", ""
    segments = entry.get("segments", [])
    transcript = format_segments(segments, "Сплошной текст")
    return entry.get("title", ""), transcript, entry.get("summary", "")
