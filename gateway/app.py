"""Entry point for the Gradio gateway UI."""

import os
import sys

GATEWAY_DIR = os.path.dirname(__file__)
if GATEWAY_DIR not in sys.path:
    sys.path.insert(0, GATEWAY_DIR)
sys.path.insert(0, os.path.join(GATEWAY_DIR, ".."))

from shared.config import GATEWAY_PID_FILE
from shared.process_singleton import singleton_process
from runtime import ensure_localhost_bypasses_proxy, logger
from monitor import shutdown_workers

ensure_localhost_bypasses_proxy()

try:
    from ui.layout import launch_ui
except ModuleNotFoundError:
    print(
        "Gradio dependency is not available. Activate the main environment:\n"
        "  .\\gradio-env\\Scripts\\Activate.ps1\n"
        "  python gateway\\app.py\n"
        "If the environment is not ready yet: pip install -r requirements.txt",
        file=sys.stderr,
        flush=True,
    )
    raise


if __name__ == "__main__":
    with singleton_process("gateway", GATEWAY_PID_FILE, "UI") as acquired:
        if not acquired:
            print("gateway/app.py is already running; exiting duplicate process.", file=sys.stderr)
            sys.exit(0)
        try:
            launch_ui()
        except Exception:
            logger.exception("Critical error while launching gateway/app.py")
            raise
        finally:
            try:
                shutdown_workers()
            except Exception:
                logger.exception("Failed to shutdown workers on gateway exit")
