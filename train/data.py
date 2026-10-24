"""Training data access: cached features + onset targets, split by song."""
import json
import zlib
from pathlib import Path

import numpy as np

from chc import features, nn

TOL = 0.05  # seconds a detection may be off and still count


def load_index(path="data/index.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_of(song: dict) -> str:
    h = zlib.crc32(song["name"].encode("utf-8")) % 100
    return "val" if h < 5 else "test" if h < 10 else "train"


def base(kind: str) -> str:
    return kind.rstrip("2")


def events(song: dict, kind: str):
    """[(time_s, channel_index)] ground truth for this instrument."""
    kind = base(kind)
    if kind == "mix":
        return [(t, 0) for t in song["beats"]] + [(t, 1) for t in song["downbeats"]]
    out = []
    for n in song["notes"][kind]:
        if kind == "drums":
            t, lane, tom = n
            ch = 0 if lane == 0 else 1 if lane == 1 else 2 + (lane - 2) * 2 + int(tom)
            out.append((t, ch))
        else:
            t, lanes = n[0], n[1]
            out += [(t, lane) for lane in lanes]
    return out


def attributes(song: dict, kind: str):
    """Guitar/bass -> ([hammer-on note times], [(start, end) of sustains])."""
    hopo, sus = [], []
    for n in song["notes"][base(kind)]:
        if len(n) < 4:
            continue
        t, _, length, h = n
        if h:
            hopo.append(t)
        if length > 0:
            sus.append((t, t + length))
    return hopo, sus


def targets(evs, n_frames: int, n_ch: int, sigma=0.8) -> np.ndarray:
    y = np.zeros((n_frames, n_ch), np.float32)
    offs = np.arange(-2, 3)
    for t, c in evs:
        f = t * features.FPS
        f0 = int(round(f))
        for o in offs:
            i = f0 + o
            if 0 <= i < n_frames:
                y[i, c] = max(y[i, c], float(np.exp(-0.5 * ((i - f) / sigma) ** 2)))
    return y


def feat_path(i: int, kind: str) -> Path:
    return Path("data/feats") / f"{i:04d}_{kind}.npy"


def dm_path(i: int, kind: str) -> Path:
    """Features of the Demucs-separated stem (what the app hears) for the same song."""
    return Path("data/feats_dm") / f"{i:04d}_{base(kind)}.npy"


def songs_for(kind: str, split: str, index):
    return [(i, s) for i, s in enumerate(index)
            if (base(kind) in s["notes"] or (kind == "mix" and s["beats"]))
            and split_of(s) == split and feat_path(i, kind).exists()]


def f1(pred, truth, n_ch, tol=TOL):
    """pred: [(time, ch)], truth same. Returns (tp, fp, fn) with greedy one-to-one matching per channel."""
    tp = fp = fn = 0
    for c in range(n_ch):
        p = sorted(t for t, k in pred if k == c)
        g = sorted(t for t, k in truth if k == c)
        used = [False] * len(p)
        j = 0
        for t in g:
            while j < len(p) and p[j] < t - tol:
                j += 1
            m = j
            while m < len(p) and p[m] <= t + tol:
                if not used[m]:
                    used[m] = True
                    tp += 1
                    break
                m += 1
            else:
                fn += 1
        fp += used.count(False)
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return p, r, 2 * p * r / max(1e-9, p + r)
