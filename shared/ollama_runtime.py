import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

from shared.config import OLLAMA_URL


def is_ollama_available(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/ps", timeout=timeout):
            return True
    except Exception:
        return False


def _ollama_process_running() -> bool:
    if psutil is None:
        return False
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if "ollama" in name or "ollama" in cmdline:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def ensure_ollama_started(wait_timeout: float = 12.0) -> Optional[subprocess.Popen]:
    """Starts `ollama serve` if the API is not reachable yet."""
    if is_ollama_available():
        return None

    if _ollama_process_running():
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            if is_ollama_available():
                return None
            time.sleep(0.5)
        return None

    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        return None

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    deadline = time.time() + wait_timeout
    while time.time() < deadline:
        if is_ollama_available():
            return proc
        if proc.poll() is not None:
            break
        time.sleep(0.5)

    return proc
