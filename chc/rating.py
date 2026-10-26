"""Difficulty tier (0-6) per instrument, as song.ini rates songs, from a chart's notes.

The tier is predicted from features a player feels (speed, busiest stretches, fast runs,
chords, hand movement, drum kicks and two-hand hits) with a small linear model learned from
the Quickplay library's own ratings (train/fit_rating.py writes models/rating.json).
"""
import json
from pathlib import Path

import numpy as np

RATING_FILE = Path(__file__).resolve().parent.parent / "models" / "rating.json"


def _density(t: np.ndarray, dur: float):
    """Notes per second: overall, and in the busiest / 90th-percentile 5-second stretch."""
    if len(t) < 2 or dur <= 0:
        return 0.0, 0.0, 0.0
    hist, _ = np.histogram(t, bins=np.arange(t[0], t[-1] + 1.0, 1.0))
    if len(hist) < 5:
        return len(t) / dur, len(t) / dur, len(t) / dur
    win = np.convolve(hist, np.ones(5), mode="valid") / 5.0
    return len(t) / dur, float(win.max()), float(np.percentile(win, 90))


def _longest_run(gaps: np.ndarray, limit: float) -> int:
    best = run = 0
    for g in gaps:
        run = run + 1 if g < limit else 0
        best = max(best, run)
    return best


def fretted_features(notes) -> dict:
    """notes: [(time, lanes[, sustain_s, hammer_on, ...])] for guitar or bass (Expert)."""
    # the fallback decoder gives a single lane number instead of a tuple of lanes
    notes = sorted(((n[0], tuple(n[1]) if isinstance(n[1], (tuple, list)) else (n[1],), *n[2:])
                    for n in notes), key=lambda n: n[0])
    t = np.array([n[0] for n in notes], float)
    if len(t) < 10:
        return {}
    dur = float(t[-1] - t[0]) or 1.0
    nps, peak, p90 = _density(t, dur)
    gaps = np.diff(t)
    lows = np.array([min(n[1]) for n in notes])
    moves = np.diff(lows)
    size = np.array([len(n[1]) for n in notes])
    return {
        "nps": nps, "peak_nps": peak, "p90_nps": p90,
        "fast": float(np.mean(gaps < 0.13)), "very_fast": float(np.mean(gaps < 0.09)),
        "longest_fast_run": np.log1p(_longest_run(gaps, 0.15)),
        "chords": float(np.mean(size > 1)), "triples": float(np.mean(size > 2)),
        "moves": float(np.mean(moves != 0)), "jump": float(np.mean(np.abs(moves))),
        "big_jumps": float(np.mean(np.abs(moves) >= 2)),
        "orange": float(np.mean([4 in n[1] for n in notes])),
        "hopo": float(np.mean([bool(n[3]) for n in notes])) if len(notes[0]) > 3 else 0.0,
        "sustain": float(np.mean([n[2] > 0 for n in notes])) if len(notes[0]) > 2 else 0.0,
    }


def drum_features(hits) -> dict:
    """hits: [(time, lane 0-4, is_tom, ...)] (Expert)."""
    hits = sorted(hits, key=lambda h: h[0])
    t = np.array([h[0] for h in hits], float)
    if len(t) < 10:
        return {}
    # moments: hits within 20 ms are played together
    starts = np.r_[True, np.diff(t) > 0.02]
    moment = np.cumsum(starts) - 1
    mt = t[starts]
    dur = float(mt[-1] - mt[0]) or 1.0
    nps, peak, p90 = _density(mt, dur)
    lanes = np.array([h[1] for h in hits])
    kicks = t[lanes == 0]
    kick_gaps = np.diff(kicks) if len(kicks) > 1 else np.array([1.0])
    hands = np.bincount(moment[lanes != 0], minlength=len(mt))
    gaps = np.diff(mt)
    return {
        "nps": nps, "peak_nps": peak, "p90_nps": p90,
        "kicks_ps": len(kicks) / dur, "fast_kicks": float(np.mean(kick_gaps < 0.15)),
        "double_kick": float(np.mean(kick_gaps < 0.11)),
        "two_hands": float(np.mean(hands >= 2)),
        "toms": float(np.mean([bool(h[2]) for h in hits if h[1] >= 2] or [0.0])),
        "fast": float(np.mean(gaps < 0.13)), "longest_fast_run": np.log1p(_longest_run(gaps, 0.15)),
        "pads_used": float(len(set((h[1], bool(h[2])) for h in hits))),
    }


def features(inst: str, notes) -> dict:
    return drum_features(notes) if inst == "drums" else fretted_features(notes)


_model = None


def tier(inst: str, notes) -> int:
    """Predicted song.ini difficulty (0-6); 2 when no rating model has been trained yet or the
    part is too short to judge."""
    global _model
    if _model is None:
        _model = json.loads(RATING_FILE.read_text()) if RATING_FILE.exists() else {}
    m = _model.get("drums" if inst == "drums" else inst)
    f = features(inst, notes)
    if not m or not f:
        return 2
    x = np.array([f[k] for k in m["features"]])
    z = (x - np.array(m["mean"])) / np.array(m["std"])
    return int(np.clip(np.round(float(z @ np.array(m["coef"]) + m["bias"])), 0, 6))
