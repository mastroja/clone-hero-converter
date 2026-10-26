"""RockWizard desktop app: starts the local server (also used by the phone web app) and shows the
interface in its own window. If the server is already running, it just opens another window."""
import os
import socket
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import webview  # noqa: E402

from chc import server  # noqa: E402

window = None


class Api:
    """Called from the page (window.pywebview.api) for things only the PC can do."""

    def pick_folder(self):
        r = window.create_file_dialog(webview.FOLDER_DIALOG)
        return r[0] if r else None

    def pick_file(self):
        r = window.create_file_dialog(webview.OPEN_DIALOG, file_types=("Programs (*.exe)", "All files (*.*)"))
        return r[0] if r else None


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    global window
    if not port_open(server.PORT):
        threading.Thread(target=server.serve, daemon=True).start()
        for _ in range(100):
            if port_open(server.PORT):
                break
            time.sleep(0.1)
    window = webview.create_window("RockWizard", f"http://127.0.0.1:{server.PORT}/", js_api=Api(),
                                   width=1200, height=820, min_size=(380, 600), background_color="#0c0d1a")
    webview.start(icon=str(ROOT / "rockwizard.ico"))


if __name__ == "__main__":
    main()
