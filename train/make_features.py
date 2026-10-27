"""python -m train.make_features [index.json] : decode stems and cache features as .npy."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from chc import audio, features

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def decode(path: str, sr: int = features.SR) -> np.ndarray:
    r = subprocess.run([audio.ffmpeg_exe(), "-v", "error", "-i", path, "-ac", "1", "-ar",
                        str(sr), "-f", "f32le", "-"],
                       capture_output=True, creationflags=NO_WINDOW)
    return np.frombuffer(r.stdout, dtype=np.float32)


def decode_stereo(path: str, sr: int = features.SR) -> np.ndarray:
    """(2, samples) float32; a mono file comes back as two equal channels."""
    r = subprocess.run([audio.ffmpeg_exe(), "-v", "error", "-i", path, "-ac", "2", "-ar",
                        str(sr), "-f", "f32le", "-"],
                       capture_output=True, creationflags=NO_WINDOW)
    return np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, 2).T


def load_stereo_stack(guitar_paths: list[str], keys_path: str, sr: int = features.SR) -> np.ndarray:
    """(3, samples) for features.STACKS['guitar_st']: middle and sides of the guitar audio
    (several files are summed), and the keys stem in mono."""
    waves = [decode_stereo(p, sr) for p in guitar_paths]
    keys = decode(keys_path, sr)
    n = max([w.shape[1] for w in waves] + [len(keys)])
    st = np.zeros((2, n), np.float32)
    for w in waves:
        st[:, :w.shape[1]] += w
    out = np.zeros((3, n), np.float32)
    out[0] = (st[0] + st[1]) / 2
    out[1] = (st[0] - st[1]) / 2
    out[2, :len(keys)] = keys
    return out


def load_stem(paths: list[str], sr: int = features.SR) -> np.ndarray:
    waves = [decode(p, sr) for p in paths]
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
