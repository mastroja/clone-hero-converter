"""Learned difficulty reduction: which Expert notes a charter keeps on Hard, Medium and Easy.

Every Expert note (a drum hit, or a guitar/bass note or chord) is described by where it sits in
the beat and bar, what it is (pad / frets), how busy its surroundings are and the tempo; a small
model per instrument and level (train/fit_reduce.py -> models/reduce.joblib), learned from the
library's real reduced charts, gives the chance it is kept.
"""
from pathlib import Path

import numpy as np

MODEL_FILE = Path(__file__).resolve().parent.parent / "models" / "reduce.joblib"
LEVELS = ("hard", "medium", "easy")


def _pos_class(frac: float) -> int:
    """0 on the beat, 1 on the 8th, 2 on a 16th, 3 on a triplet, 4 anything finer."""
    for c, points in ((0, (0.0, 1.0)), (1, (0.5,)), (2, (0.25, 0.75)), (3, (1 / 3, 2 / 3))):
        if any(abs(frac - p) < 0.02 for p in points):
            return c
    return 4


def _onehot(i: int, n: int):
    v = [0.0] * n
    v[i] = 1.0
    return v


def drum_features(hits, tpq: int, spb) -> np.ndarray:
    """hits: [(tick, lane, tom)] (Expert, sorted); spb(tick): seconds per beat there."""
    ticks = np.array([h[0] for h in hits], float)
    lanes = np.array([h[1] for h in hits])
    moments = np.unique(ticks)
    density = len(moments) / max(1.0, (ticks[-1] - ticks[0]) / tpq) if len(ticks) else 0.0
    by_lane = {l: np.sort(ticks[lanes == l]) for l in range(5)}
    at = {}
    for t, l in zip(ticks, lanes):
        at.setdefault(t, []).append(l)
    rows = []
    for t, l, tom in hits:
        same = by_lane[l]
        k = np.searchsorted(same, t)
        prev_same = (t - same[k - 1]) / tpq if k > 0 else 4.0
        next_same = (same[k + 1] - t) / tpq if k + 1 < len(same) else 4.0
        m = np.searchsorted(moments, t)
        prev_any = (t - moments[m - 1]) / tpq if m > 0 else 4.0
        next_any = (moments[m + 1] - t) / tpq if m + 1 < len(moments) else 4.0
        here = at[t]
        rows.append(_onehot(l, 5) + [float(tom)]
                    + _onehot(_pos_class((t % tpq) / tpq), 5) + _onehot(int(t // tpq) % 4, 4)
                    + [float(np.sum(np.abs(same - t) <= tpq)),
                       float(np.sum(np.abs(moments - t) <= tpq)),
                       float(len(here)), float(0 in here and l != 0), float(1 in here and l != 1),
                       min(prev_same, 4.0), min(next_same, 4.0), min(prev_any, 4.0), min(next_any, 4.0),
                       spb(t), density])
    return np.array(rows, np.float32)


def fret_features(notes, tpq: int, spb) -> np.ndarray:
    """notes: [(tick, lanes)] (Expert, sorted, one entry per note or chord)."""
    ticks = np.array([n[0] for n in notes], float)
    density = len(ticks) / max(1.0, (ticks[-1] - ticks[0]) / tpq) if len(ticks) else 0.0
    rows = []
    for k, (t, lanes) in enumerate(notes):
        prev = (t - ticks[k - 1]) / tpq if k > 0 else 4.0
        nxt = (ticks[k + 1] - t) / tpq if k + 1 < len(ticks) else 4.0
        lo = min(lanes)
        prev_lo = min(notes[k - 1][1]) if k > 0 else lo
        rows.append([float(len(lanes)), float(lo), float(max(lanes))]
                    + _onehot(_pos_class((t % tpq) / tpq), 5) + _onehot(int(t // tpq) % 4, 4)
                    + [min(prev, 4.0), min(nxt, 4.0), float(np.sum(np.abs(ticks - t) <= tpq)),
                       float(np.sum(np.abs(ticks - t) <= tpq / 2)),
                       float(k > 0 and tuple(notes[k - 1][1]) == tuple(lanes)), float(lo - prev_lo),
                       spb(t), density])
    return np.array(rows, np.float32)


_models = None


def models():
    global _models
    if _models is None:
        _models = {}
        if MODEL_FILE.exists():
            import joblib
            _models = joblib.load(MODEL_FILE)
    return _models


def keep(inst: str, level: str, notes, tpq: int, spb):
    """Boolean per Expert note: kept on this level. None when no model is available."""
    m = models().get((inst, level))
    if m is None or not notes:
        return None
    X = drum_features(notes, tpq, spb) if inst == "drums" else fret_features(notes, tpq, spb)
    return m["model"].predict_proba(X)[:, 1] >= m["threshold"]
