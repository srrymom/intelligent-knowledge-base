"""SQLite registry for user processing tasks."""

import glob
import json
import os
import sqlite3
from datetime import datetime

from shared.config import (
    DATA_DIR,
    KB_DIR,
    QUEUE_DIR,
    SUMMARY_DIR,
    TRANSCRIPT_DIR,
)

TASKS_DB_FILE = os.path.join(DATA_DIR, "tasks.sqlite3")
QUEUE_PROCESSING_DIR = os.path.join(DATA_DIR, "queue_processing")


STATUS_LABELS = {
    "queued_asr": "Ожидает транскрипции",
    "asr_running": "Транскрипция",
    "waiting_llm": "Ожидает суммаризации",
    "llm_running": "Суммаризация",
    "done": "Готово",
    "error": "Ошибка",
    "missing": "Файлы не найдены",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _connect():
    conn = sqlite3.connect(TASKS_DB_FILE, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_name TEXT NOT NULL DEFAULT '',
                report_mode TEXT NOT NULL DEFAULT '',
                sum_method TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                status_label TEXT NOT NULL,
                progress_stage TEXT,
                progress_current INTEGER,
                progress_total INTEGER,
                result_title TEXT,
                kb_id TEXT,
                error TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")


init_db()


def _row_to_dict(row) -> dict | None:
    return dict(row) if row is not None else None


def create_task(
    task_id: str,
    *,
    source_type: str,
    source_name: str,
    report_mode: str,
    sum_method: str,
    status: str,
) -> None:
    label = STATUS_LABELS.get(status, status)
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO tasks (
                id, created_at, updated_at, source_type, source_name,
                report_mode, sum_method, status, status_label,
                progress_stage, progress_current, progress_total,
                result_title, kb_id, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL)
            """,
            (
                task_id,
                now,
                now,
                source_type,
                source_name or "",
                report_mode or "",
                sum_method or "",
                status,
                label,
            ),
        )


def get_task(task_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _row_to_dict(row)


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _has_uuid_file(directory: str, task_id: str) -> bool:
    return bool(glob.glob(os.path.join(directory, f"{task_id}.*")))


def inspect_task(task: dict) -> dict:
    task_id = task["id"]
    status = task.get("status") or "missing"
    result_title = task.get("result_title")
    kb_id = task.get("kb_id")
    error = task.get("error")
    progress_stage = None
    progress_current = None
    progress_total = None

    kb_path = os.path.join(KB_DIR, f"{task_id}.json")
    if os.path.exists(kb_path):
        entry = _read_json(kb_path) or {}
        result_title = entry.get("title") or result_title
        kb_id = task_id
        summary = entry.get("summary", "")
        if result_title == "Ошибка обработки" or summary.startswith("Ошибка обработки:"):
            status = "error"
            error = summary or "Ошибка обработки"
        else:
            status = "done"
            error = None
        progress_stage = "done"
        progress_current = 1
        progress_total = 1
    else:
        llm_progress_path = os.path.join(SUMMARY_DIR, f"{task_id}.progress")
        asr_progress_path = os.path.join(SUMMARY_DIR, f"{task_id}.asr.progress")
        transcript_path = os.path.join(TRANSCRIPT_DIR, f"{task_id}.json")
        transcript_lock_path = os.path.join(TRANSCRIPT_DIR, f"{task_id}.json.lock")

        if os.path.exists(llm_progress_path):
            prog = _read_json(llm_progress_path) or {}
            status = "llm_running"
            progress_stage = prog.get("stage")
            progress_current = int(prog.get("current") or 0)
            progress_total = int(prog.get("total") or 0)
            error = None
        elif os.path.exists(asr_progress_path):
            prog = _read_json(asr_progress_path) or {}
            status = "asr_running"
            progress_stage = "asr"
            progress_current = int(prog.get("current") or 0)
            progress_total = int(prog.get("total") or 0)
            error = None
        elif os.path.exists(transcript_lock_path):
            status = "llm_running"
            progress_stage = "llm"
            error = None
        elif os.path.exists(transcript_path):
            status = "waiting_llm"
            progress_stage = "waiting_llm"
            error = None
        elif _has_uuid_file(QUEUE_PROCESSING_DIR, task_id):
            status = "asr_running"
            progress_stage = "asr"
            error = None
        elif _has_uuid_file(QUEUE_DIR, task_id):
            status = "queued_asr"
            progress_stage = "queued_asr"
            error = None
        elif status not in ("done", "error"):
            status = "missing"
            progress_stage = "missing"
            error = "Не найдены рабочие файлы задания"

    return {
        **task,
        "updated_at": _now(),
        "status": status,
        "status_label": STATUS_LABELS.get(status, status),
        "progress_stage": progress_stage,
        "progress_current": progress_current,
        "progress_total": progress_total,
        "result_title": result_title,
        "kb_id": kb_id,
        "error": error,
    }


def _persist_runtime(task: dict) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE tasks
               SET updated_at = ?,
                   status = ?,
                   status_label = ?,
                   progress_stage = ?,
                   progress_current = ?,
                   progress_total = ?,
                   result_title = ?,
                   kb_id = ?,
                   error = ?
             WHERE id = ?
            """,
            (
                task["updated_at"],
                task["status"],
                task["status_label"],
                task.get("progress_stage"),
                task.get("progress_current"),
                task.get("progress_total"),
                task.get("result_title"),
                task.get("kb_id"),
                task.get("error"),
                task["id"],
            ),
        )


def list_tasks(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def list_tasks_with_runtime(limit: int = 50) -> list[dict]:
    result = []
    for task in list_tasks(limit):
        runtime_task = inspect_task(task)
        _persist_runtime(runtime_task)
        result.append(runtime_task)
    return result


def delete_task(task_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))


def clear_finished_tasks() -> int:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM tasks WHERE status IN ('done', 'error', 'missing')")
        return cur.rowcount
