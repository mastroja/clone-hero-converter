"""Start the RockWizard server when you sign in to Windows: a shortcut in your Startup folder
(no admin rights needed; remove it from Settings or by deleting the shortcut)."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHORTCUT = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "RockWizard server.lnk"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def enabled() -> bool:
    return SHORTCUT.exists()


def enable() -> None:
    target = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    script = ROOT / "rockwizard_server.pyw"
    q = lambda s: str(s).replace("'", "''")  # PowerShell single-quoted string
    ps = (f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{q(SHORTCUT)}'); "
          f"$s.TargetPath = '{q(target)}'; $s.Arguments = '\"{q(script)}\"'; "
          f"$s.WorkingDirectory = '{q(ROOT)}'; $s.IconLocation = '{q(ROOT / 'rockwizard.ico')}'; "
          f"$s.Description = 'RockWizard server (for the phone app)'; $s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True,
                   creationflags=NO_WINDOW)


def disable() -> None:
    SHORTCUT.unlink(missing_ok=True)
