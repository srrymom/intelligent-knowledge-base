"""Runtime setup for the Gradio gateway."""

import os
import shutil
import subprocess
import sys
from html import escape

from shared.config import (
    ACTIVITY_LOG,
    DATA_DIR,
    FFMPEG_PATH,
    KB_DIR,
    PROJECT_ROOT,
    QUEUE_DIR,
    RAG_DB_DIR,
    SUMMARY_DIR,
    TRANSCRIPT_DIR,
)
from shared.gpu_coord import reset_gpu_state
from shared.log import read_tail, setup_runtime_logging, write_resource_event
from shared.ollama_runtime import ensure_ollama_started
from monitor import register_workers

logger = setup_runtime_logging("UI")


def ensure_localhost_bypasses_proxy():
    local_hosts = ["localhost", "127.0.0.1", "::1"]
    for env_name in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(env_name, "")
        parts = [p.strip() for p in current.split(",") if p.strip()]
        for host in local_hosts:
            if host not in parts:
                parts.append(host)
        os.environ[env_name] = ",".join(parts)


def format_activity_log(lines: int = 120) -> str:
    text = read_tail(lines).strip()
    if not text:
        text = "Журнал пока пуст."
    return f'<pre class="activity-log">{escape(text)}</pre>'


def start_workers():
    logger.info("Сброс состояния GPU-координации")
    logger.info(
        "DIAG start_workers pid=%s ppid=%s exe=%s argv=%s",
        os.getpid(),
        os.getppid(),
        sys.executable,
        " ".join(sys.argv),
    )
    write_resource_event("UI", "Перед запуском воркеров")
    reset_gpu_state()
    logger.info("Проверяю Ollama: %s", os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ollama_proc = ensure_ollama_started()
    if ollama_proc is None:
        logger.info("Ollama уже доступна или будет ожидаться воркерами")
    else:
        logger.info("Запущен subprocess Ollama serve, pid=%s", ollama_proc.pid)

    venv = "Scripts" if sys.platform == "win32" else "bin"
    python_name = "python.exe" if sys.platform == "win32" else "python"
    asr_python = os.path.join(PROJECT_ROOT, "asr", ".venv", venv, python_name)
    asr_worker = os.path.join(PROJECT_ROOT, "asr", "worker.py")
    llm_python = os.path.join(PROJECT_ROOT, "llm", ".venv", venv, python_name)
    llm_worker = os.path.join(PROJECT_ROOT, "llm", "worker.py")

    missing = [
        path for path in (asr_python, asr_worker, llm_python, llm_worker)
        if not os.path.exists(path)
    ]
    if missing:
        for path in missing:
            logger.error("Не найден обязательный файл запуска: %s", path)
        raise FileNotFoundError(
            "Не найдены файлы для запуска воркеров. "
            "Проверь установку asr/.venv и llm/.venv."
        )

    logger.info("Стартую ASR worker: %s %s", asr_python, asr_worker)
    asr_proc = subprocess.Popen([asr_python, asr_worker])
    logger.info("ASR worker pid=%s", asr_proc.pid)
    logger.info("Стартую LLM worker: %s %s", llm_python, llm_worker)
    llm_proc = subprocess.Popen([llm_python, llm_worker])
    logger.info("LLM worker pid=%s", llm_proc.pid)
    register_workers(asr_proc, llm_proc)
    write_resource_event("UI", "Воркеры запущены")
    return asr_proc, llm_proc

def log_startup_environment():
    logger.info("Запуск gateway/app.py")
    logger.info("Python: %s", sys.executable)
    logger.info("Версия Python: %s", sys.version.split()[0])
    logger.info("Платформа: %s", sys.platform)
    logger.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logger.info("DATA_DIR: %s", DATA_DIR)
    logger.info("ACTIVITY_LOG: %s", ACTIVITY_LOG)
    logger.info("Рабочие папки: queue=%s transcript=%s summary=%s kb=%s rag=%s",
                QUEUE_DIR, TRANSCRIPT_DIR, SUMMARY_DIR, KB_DIR, RAG_DB_DIR)
    logger.info("FFMPEG_PATH: %s", FFMPEG_PATH)
    logger.info("ffmpeg в PATH: %s", shutil.which("ffmpeg") or "не найден")
    logger.info("ollama в PATH: %s", shutil.which("ollama") or "не найден")
    write_resource_event("UI", "Старт приложения")
