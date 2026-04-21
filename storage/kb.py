"""
CRUD для базы знаний. Каждая запись -- UUID.json в KB_DIR: транскрипт, конспект, темы. Список, чтение, удаление, статистика, список уникальных тем. При удалении синхронно чистит чанки из ChromaDB.
"""

import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import KB_DIR, PROJECT_ROOT

sys.path.insert(0, os.path.join(PROJECT_ROOT, "rag"))
from engine import remove_entry


def _load_all_entries():
    entries = []
    for fname in os.listdir(KB_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(KB_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
    return entries


def load_kb_list(topic_filter=None):
    entries = _load_all_entries()

    if topic_filter:
        entries = [e for e in entries if topic_filter in e.get("topics", [])]

    table_data = []
    uuid_list = []
    for e in entries:
        date_str = e.get("created_at", "")[:16].replace("T", " ")
        table_data.append([date_str, e.get("title", "(без названия)")])
        uuid_list.append(e["id"])

    return table_data, uuid_list


def get_all_topics():
    """Возвращает список уникальных тем из всей базы знаний, отсортированных по частоте."""
    from collections import Counter
    counter = Counter()
    for e in _load_all_entries():
        for topic in e.get("topics", []):
            counter[topic] += 1
    return [t for t, _ in counter.most_common()]


def load_kb_entry(entry_id):
    path = os.path.join(KB_DIR, f"{entry_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_kb_entry(entry_id):
    path = os.path.join(KB_DIR, f"{entry_id}.json")
    if os.path.exists(path):
        os.remove(path)
    # удалить чанки из RAG-индекса
    remove_entry(entry_id)


def get_kb_stats():
    """Возвращает (count, total_words, total_audio_sec) по всей базе знаний."""
    count = 0
    total_words = 0
    total_audio_sec = 0
    for fname in os.listdir(KB_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(KB_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            count += 1
            for seg in data.get("segments", []):
                text = seg.get("transcription", "")
                if text.replace("⁇", "").replace(" ", "").strip():
                    total_words += len(text.split())
                b = seg.get("boundaries", [0, 0])
                if b[1] > b[0]:
                    total_audio_sec += b[1] - b[0]
        except Exception:
            continue
    return count, total_words, int(total_audio_sec)
