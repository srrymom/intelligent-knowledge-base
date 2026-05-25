"""Small pid-file guard for one live process per role."""

import json
import os
import sys
from contextlib import contextmanager

from shared.log import write_event


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False

    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid(lock_path: str) -> int | None:
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("pid", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def get_live_singleton_pid(lock_path: str) -> int | None:
    pid = _read_pid(lock_path)
    if pid and _pid_is_alive(pid):
        return pid
    return None


def _write_lock(lock_path: str, role: str) -> bool:
    payload = {
        "role": role,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "exe": sys.executable,
        "argv": sys.argv,
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(lock_path, flags)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return True


def _remove_own_lock(lock_path: str) -> None:
    if _read_pid(lock_path) != os.getpid():
        return
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


@contextmanager
def singleton_process(role: str, lock_path: str, log_source: str):
    """Yield False when another live process already owns this role."""
    while not _write_lock(lock_path, role):
        existing_pid = _read_pid(lock_path)
        if existing_pid and _pid_is_alive(existing_pid):
            write_event(
                log_source,
                f"DIAG SINGLETON_EXISTS role={role} existing_pid={existing_pid} "
                f"current_pid={os.getpid()} lock={lock_path}",
            )
            yield False
            return
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass

    write_event(log_source, f"DIAG SINGLETON_ACQUIRED role={role} pid={os.getpid()} lock={lock_path}")
    try:
        yield True
    finally:
        _remove_own_lock(lock_path)
        write_event(log_source, f"DIAG SINGLETON_RELEASED role={role} pid={os.getpid()} lock={lock_path}")
