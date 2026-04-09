import json
import os
import time

from shared.config import GPU_STATE_FILE


def _default_state() -> dict:
    return {"owner": None, "requests": {"asr": False, "llm": False}, "updated_at": 0}


def read_gpu_state() -> dict:
    try:
        with open(GPU_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "owner": data.get("owner"),
            "requests": {
                "asr": bool((data.get("requests") or {}).get("asr", False)),
                "llm": bool((data.get("requests") or {}).get("llm", False)),
            },
            "updated_at": data.get("updated_at", 0),
        }
    except Exception:
        return _default_state()


def _write_gpu_state(state: dict) -> None:
    tmp_path = f"{GPU_STATE_FILE}.tmp"
    payload = {
        "owner": state.get("owner"),
        "requests": state.get("requests", {"asr": False, "llm": False}),
        "updated_at": time.time(),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, GPU_STATE_FILE)


def request_gpu(worker: str) -> None:
    state = read_gpu_state()
    requests = state.setdefault("requests", {"asr": False, "llm": False})
    if requests.get(worker):
        return
    requests[worker] = True
    _write_gpu_state(state)


def clear_gpu_request(worker: str) -> None:
    state = read_gpu_state()
    requests = state.setdefault("requests", {"asr": False, "llm": False})
    if not requests.get(worker):
        return
    requests[worker] = False
    _write_gpu_state(state)


def acquire_gpu(worker: str) -> bool:
    state = read_gpu_state()
    owner = state.get("owner")
    if owner not in (None, worker):
        return False
    state["owner"] = worker
    state.setdefault("requests", {"asr": False, "llm": False})[worker] = False
    _write_gpu_state(state)
    return True


def release_gpu(worker: str) -> None:
    state = read_gpu_state()
    if state.get("owner") != worker:
        return
    state["owner"] = None
    _write_gpu_state(state)


def reset_gpu_state() -> None:
    _write_gpu_state(_default_state())
