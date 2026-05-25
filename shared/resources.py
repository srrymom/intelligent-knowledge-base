"""
Короткие снимки ресурсов для диагностических логов.
"""

import os
import subprocess

try:
    import psutil
except ImportError:
    psutil = None


def _gb(value_bytes: int | float) -> float:
    return round(value_bytes / (1024**3), 1)


def _current_process_ram_mb() -> int | None:
    if psutil is None:
        return None
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except Exception:
        return None


def get_resource_snapshot() -> dict:
    snapshot = {
        "pid": os.getpid(),
        "process_ram_mb": _current_process_ram_mb(),
        "cpu_pct": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
        "ram_pct": None,
        "gpu_available": False,
    }

    if psutil is not None:
        try:
            mem = psutil.virtual_memory()
            snapshot.update(
                {
                    "cpu_pct": psutil.cpu_percent(interval=0),
                    "ram_used_gb": _gb(mem.used),
                    "ram_total_gb": _gb(mem.total),
                    "ram_pct": mem.percent,
                }
            )
        except Exception:
            pass

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        snapshot.update(
            {
                "gpu_available": True,
                "gpu_name": parts[0],
                "vram_total_mb": int(parts[1]),
                "vram_used_mb": int(parts[2]),
                "vram_free_mb": int(parts[3]),
                "gpu_util_pct": int(parts[4]),
                "gpu_temp_c": int(parts[5]),
            }
        )
    except Exception:
        pass

    return snapshot


def format_resource_snapshot(snapshot: dict | None = None) -> str:
    data = snapshot or get_resource_snapshot()
    parts = [f"pid={data.get('pid')}"]

    proc_ram = data.get("process_ram_mb")
    if proc_ram is not None:
        parts.append(f"proc_ram={proc_ram}MB")

    if data.get("cpu_pct") is not None:
        parts.append(f"cpu={data['cpu_pct']}%")

    if data.get("ram_used_gb") is not None:
        parts.append(
            f"ram={data['ram_used_gb']}/{data['ram_total_gb']}GB ({data['ram_pct']}%)"
        )

    if data.get("gpu_available"):
        total = data["vram_total_mb"]
        used = data["vram_used_mb"]
        vram_pct = round(used / total * 100) if total else 0
        parts.append(
            f"gpu={data['gpu_name']} vram={used}/{total}MB ({vram_pct}%) "
            f"util={data['gpu_util_pct']}% temp={data['gpu_temp_c']}C"
        )
    else:
        parts.append("gpu=not_available")

    return " | ".join(parts)
