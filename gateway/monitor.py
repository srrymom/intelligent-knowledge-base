import os
import sys
import subprocess
import psutil
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import LOCK_FILE, OLLAMA_URL, PROJECT_ROOT

# Popen-объекты воркеров — регистрируются из app.py при старте
_workers: dict = {}


def register_workers(asr_proc, llm_proc):
    _workers["asr"] = {"proc": asr_proc, "label": "ASR Worker", "script": "asr/worker.py"}
    _workers["llm"] = {"proc": llm_proc, "label": "LLM Worker", "script": "llm/worker.py"}


def get_worker_health() -> list[dict]:
    """Проверяет состояние каждого воркера. При падении — перезапускает."""
    results = []
    for key, info in _workers.items():
        proc = info["proc"]
        alive = proc.poll() is None  # None = процесс жив
        if not alive:
            # Перезапуск
            venv = "Scripts" if sys.platform == "win32" else "bin"
            python = os.path.join(PROJECT_ROOT, key, ".venv", venv, "python")
            script = os.path.join(PROJECT_ROOT, info["script"])
            new_proc = subprocess.Popen([python, script])
            info["proc"] = new_proc
            status = "перезапущен"
        else:
            status = "работает"
        results.append({"label": info["label"], "alive": alive, "status": status})
    return results


def get_gpu_stats():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "available": True,
            "name": parts[0],
            "vram_total_mb": int(parts[1]),
            "vram_used_mb": int(parts[2]),
            "vram_free_mb": int(parts[3]),
            "gpu_util_pct": int(parts[4]),
            "temp_c": int(parts[5]),
        }
    except Exception:
        return {"available": False}


def get_cpu_stats():
    mem = psutil.virtual_memory()
    return {
        "cpu_pct": psutil.cpu_percent(interval=0),
        "ram_used_gb": round(mem.used / (1024**3), 1),
        "ram_total_gb": round(mem.total / (1024**3), 1),
        "ram_pct": mem.percent,
    }


def get_ollama_status():
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/ps", timeout=2)
        data = resp.json()
        models = data.get("models", [])
        if models:
            m = models[0]
            size_vram_mb = round(m.get("size_vram", 0) / (1024**2))
            size_total_mb = round(m.get("size", 0) / (1024**2))
            details = m.get("details", {})
            return {
                "loaded": True,
                "model": m.get("model", "?"),
                "params": details.get("parameter_size", "?"),
                "quant": details.get("quantization_level", "?"),
                "size_total_mb": size_total_mb,
                "size_vram_mb": size_vram_mb,
            }
        return {"loaded": False}
    except Exception:
        return {"loaded": False, "error": "Ollama недоступен"}


def _find_asr_process():
    for proc in psutil.process_iter(["pid", "cmdline", "memory_info"]):
        try:
            cmdline = proc.info["cmdline"]
            if cmdline and any("asr" in arg and "worker.py" in arg for arg in cmdline):
                mem = proc.info["memory_info"]
                return {"ram_mb": round(mem.rss / (1024**2))}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def get_asr_status():
    loaded = os.path.exists(LOCK_FILE)
    proc = _find_asr_process()
    return {"loaded": loaded, "process": proc}


def format_status():
    gpu = get_gpu_stats()
    cpu = get_cpu_stats()
    llm_status = get_ollama_status()
    asr = get_asr_status()
    worker_health = get_worker_health()

    lines = []

    if worker_health:
        for w in worker_health:
            icon = "🟢" if w["alive"] else "🔴"
            lines.append(f"{icon} {w['label']}: {w['status']}")
        lines.append("")

    if gpu["available"]:
        vram_pct = round(gpu["vram_used_mb"] / gpu["vram_total_mb"] * 100)
        lines.append(f"GPU: {gpu['name']}")
        lines.append(
            f"  VRAM: {gpu['vram_used_mb']} / {gpu['vram_total_mb']} МБ ({vram_pct}%)"
        )
        lines.append(f"  Загрузка: {gpu['gpu_util_pct']}%  |  Темп: {gpu['temp_c']}°C")
    else:
        lines.append("GPU: не обнаружен")

    lines.append("")

    lines.append(
        f"CPU: {cpu['cpu_pct']}%  |  "
        f"RAM: {cpu['ram_used_gb']} / {cpu['ram_total_gb']} ГБ ({cpu['ram_pct']}%)"
    )

    lines.append("")

    if llm_status.get("loaded"):
        lines.append(
            f"LLM (Qwen): загружена  —  {llm_status['model']} "
            f"({llm_status['params']}, {llm_status['quant']})"
        )
        lines.append(
            f"  VRAM модели: {llm_status['size_vram_mb']} МБ "
            f"(всего: {llm_status['size_total_mb']} МБ)"
        )
    elif llm_status.get("error"):
        lines.append(f"LLM (Qwen): {llm_status['error']}")
    else:
        lines.append("LLM (Qwen): выгружена")

    if asr["loaded"]:
        line = "ASR (GigaAM): загружена (идёт транскрипция)"
        if asr.get("process"):
            line += f"\n  RAM процесса: {asr['process']['ram_mb']} МБ"
        if gpu["available"] and llm_status.get("loaded"):
            asr_vram = gpu["vram_used_mb"] - llm_status["size_vram_mb"] - 1200
            if asr_vram > 0:
                line += f"\n  VRAM (оценка): ~{asr_vram} МБ"
        lines.append(line)
    else:
        proc = asr.get("process")
        if proc:
            lines.append(f"ASR (GigaAM): выгружена (воркер: {proc['ram_mb']} МБ RAM)")
        else:
            lines.append("ASR (GigaAM): выгружена")

    return "\n".join(lines)
