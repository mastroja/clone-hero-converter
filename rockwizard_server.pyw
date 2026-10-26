"""RockWizard server with no window, for the phone web app and for starting with Windows.
The desktop window (rockwizard.pyw) uses this server when it's already running, so closing
the window doesn't cut off the phone. Output goes to logs/rockwizard-server.log."""
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
(ROOT / "logs").mkdir(exist_ok=True)
log = open(ROOT / "logs" / "rockwizard-server.log", "a", encoding="utf-8", buffering=1)
sys.stdout = sys.stderr = log  # pythonw has no console to write to
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from chc import server  # noqa: E402


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


if __name__ == "__main__":
    if port_open(server.PORT):
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "already running on port", server.PORT)
        sys.exit(0)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "starting on port", server.PORT)
    server.serve()
