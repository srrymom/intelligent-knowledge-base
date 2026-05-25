"""
Простой append-лог активности воркеров в activity.log. Ротация при превышении MAX_LINES строк. UI читает хвост через read_tail().
"""

import logging
import os
import sys
from datetime import datetime
from shared.config import ACTIVITY_LOG
from shared.resources import format_resource_snapshot

MAX_LINES = 500  # ротация: не даём файлу расти бесконечно


def _console_enabled() -> bool:
    return os.environ.get("LOG_TO_CONSOLE", "1").lower() not in {"0", "false", "no"}


def setup_runtime_logging(source: str = "APP", level: int = logging.INFO) -> logging.Logger:
    """Настраивает понятный лог в консоль и data/activity.log для стартующих процессов."""
    logger = logging.getLogger(source)
    logger.setLevel(level)
    logger.propagate = False

    if getattr(logger, "_project_logging_ready", False):
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(name)s: %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if _console_enabled():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    try:
        file_handler = logging.FileHandler(ACTIVITY_LOG, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        pass

    logger._project_logging_ready = True
    return logger


def write_event(source: str, message: str, *, console: bool = True) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {source}: {message}\n"
    if console and _console_enabled():
        try:
            print(line, end="", flush=True)
        except OSError:
            pass
    try:
        with open(ACTIVITY_LOG, "a", encoding="utf-8") as f:
            f.write(line)
        _trim_log()
    except OSError:
        pass  # не роняем воркер из-за лога


def write_resource_event(source: str, reason: str, *, console: bool = True) -> None:
    write_event(source, f"{reason} | resources: {format_resource_snapshot()}", console=console)


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
