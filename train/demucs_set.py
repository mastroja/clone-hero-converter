"""python -m train.demucs_set <split> [limit] : separate full mixes so models can be tested on
the same kind of stems the app produces (Demucs output), not just the clean originals."""
import subprocess
import sys
from pathlib import Path

from chc import audio
from . import data

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    index = data.load_index()
    todo = [(i, s) for i, s in enumerate(index) if data.split_of(s) == split and "song" in s["stems"]]
    root = Path("data/demucs")
    for i, s in todo[:limit]:
        out = root / f"{i:04d}"
        if (out / "other.wav").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        wav = audio.to_wav(Path(s["stems"]["song"][0]), out / "input.wav")
        r = subprocess.run([sys.executable, "-m", "chc.worker", str(wav), str(out), "htdemucs_ft"],
                           capture_output=True, creationflags=NO_WINDOW)
        wav.unlink(missing_ok=True)
        print(i, s["name"], "ok" if r.returncode == 0 else "FAILED", flush=True)


if __name__ == "__main__":
    main()
