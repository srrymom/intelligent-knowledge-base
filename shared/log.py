"""
Простой append-лог активности воркеров в activity.log. Ротация при превышении MAX_LINES строк. UI читает хвост через read_tail().
"""

import os
from datetime import datetime
from shared.config import ACTIVITY_LOG

MAX_LINES = 500  # ротация: не даём файлу расти бесконечно


def write_event(source: str, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {source}: {message}\n"
    try:
        with open(ACTIVITY_LOG, "a", encoding="utf-8") as f:
            f.write(line)
        _trim_log()
    except OSError:
        pass  # не роняем воркер из-за лога


def read_tail(n: int = 10) -> str:
    try:
        with open(ACTIVITY_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except OSError:
        return ""


def _trim_log() -> None:
    try:
        with open(ACTIVITY_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > MAX_LINES:
            with open(ACTIVITY_LOG, "w", encoding="utf-8") as f:
                f.writelines(lines[-MAX_LINES:])
    except OSError:
        pass
