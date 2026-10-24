"""python -m train.make_features [index.json] : decode stems and cache features as .npy."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from chc import audio, features

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def decode(path: str) -> np.ndarray:
    r = subprocess.run([audio.ffmpeg_exe(), "-v", "error", "-i", path, "-ac", "1", "-ar",
                        str(features.SR), "-f", "f32le", "-"],
                       capture_output=True, creationflags=NO_WINDOW)
    return np.frombuffer(r.stdout, dtype=np.float32)


def load_stem(paths: list[str]) -> np.ndarray:
    waves = [decode(p) for p in paths]
    n = max(len(w) for w in waves)
    mix = np.zeros(n, np.float32)
    for w in waves:
        mix[:len(w)] += w
    return mix


def main():
    idx_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/index.json")
    songs = json.loads(idx_path.read_text(encoding="utf-8"))
    kinds = sys.argv[2].split(",") if len(sys.argv) > 2 else ["guitar", "bass", "drums"]
    out = Path("data/feats")
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, s in enumerate(songs):
        for kind in kinds:
            b = "mix" if kind == "mix" else kind.rstrip("2")
            if (b in s["notes"] or b == "mix") and b in s["stems"]:
                dest = out / f"{i:04d}_{kind}.npy"
                if not dest.exists():
                    jobs.append((i, kind, s["stems"][b], dest))
    print(len(jobs), "feature files to build", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        for k in range(0, len(jobs), 24):  # bounded batches keep memory flat
            batch = jobs[k:k + 24]
            for (i, kind, _, dest), wave in zip(batch, ex.map(lambda j: load_stem(j[2]), batch)):
                if len(wave) < features.SR:
                    continue
                np.save(dest, features.compute(wave, kind))
                done += 1
            print(done, "/", len(jobs), flush=True)
    print("done", done)


if __name__ == "__main__":
    main()
