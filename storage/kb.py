import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import KB_DIR, PROJECT_ROOT

sys.path.insert(0, os.path.join(PROJECT_ROOT, "rag"))
from engine import remove_entry


def load_kb_list():
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

    table_data = []
    uuid_list = []
    for e in entries:
        date_str = e.get("created_at", "")[:16].replace("T", " ")
        table_data.append([date_str, e.get("title", "(без названия)")])
        uuid_list.append(e["id"])

    return table_data, uuid_list


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
