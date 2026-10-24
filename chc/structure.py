"""Repeated sections: find bars that are the same music and chart them the same way.

Human charters chart a chorus once and copy it. The note models judge every moment on its own,
so each repeat comes out slightly different. Here every bar becomes a fingerprint of how the
instrument sounds on each 16th of the bar; bars with near-identical fingerprints (as part of a
run of matching bars, so one lookalike bar is not enough) are grouped, the notes of a group
vote position by position, and the winning pattern is written into every bar of the group.
Voting across repeats also outvotes stray notes that turn up in only one of them.
"""
from collections import Counter, defaultdict

import numpy as np

from . import features
from .chart import TPQ

BAR = 4 * TPQ
SLOTS = 16  # fingerprint resolution: 16ths
# how alike two bars must sound to be charted alike; tuned on validation songs so repeats match
# about as often as in real charts without costing accuracy (train.eval_chart)
REPEAT_THR = {"guitar": 0.94, "bass": 0.96, "drums": 0.95}


def bar_fingerprints(feats: np.ndarray, beatmap, n_bars: int, bands: int = 24) -> np.ndarray:
    """(n_bars, SLOTS * bands) unit vectors: the instrument's spectrum on each 16th of a bar."""
    f = features.normalize(feats)
    edges = np.linspace(0, f.shape[1], bands + 1).astype(int)
    fb = np.stack([f[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:])], 1)
    ticks = np.arange(n_bars * SLOTS + 1) * (BAR // SLOTS)
    frames = np.round(np.array([beatmap.time(t) for t in ticks]) * features.FPS).astype(int)
    frames = np.clip(frames, 0, len(fb))
    V = np.zeros((n_bars * SLOTS, bands))
    for k in range(n_bars * SLOTS):
        a, b = frames[k], max(frames[k] + 1, frames[k + 1])
        if a < len(fb):
            V[k] = fb[a:b].mean(0)
    V = V.reshape(n_bars, -1)
    V -= V.mean(1, keepdims=True)
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)


def repeat_groups(V: np.ndarray, thr: float = 0.85) -> list[list[int]]:
    """Groups of bars that repeat the same music. The first bar of each group leads it; a bar
    joins the leader it matches best, if it matches it at or above thr and a neighbouring bar
    also matches the leader's neighbour (so the bars belong to a repeated passage)."""
    S = V @ V.T
    n = len(S)
    near = thr - 0.05
    leader_of, leaders = {}, []
    for i in range(n):
        best, arg = -1.0, None
        for L in leaders:
            if S[i, L] < thr or S[i, L] <= best:
                continue
            run = (i > 0 and L > 0 and S[i - 1, L - 1] >= near) or \
                  (i + 1 < n and L + 1 < n and S[i + 1, L + 1] >= near)
            if run:
                best, arg = S[i, L], L
        if arg is None:
            leaders.append(i)
            leader_of[i] = i
        else:
            leader_of[i] = arg
    groups = defaultdict(list)
    for i, L in leader_of.items():
        groups[L].append(i)
    return [g for g in groups.values() if len(g) > 1]


def _bars(notes, beatmap):
    by = defaultdict(list)
    for n in notes:
        tick = beatmap.tick(n[0])
        by[tick // BAR].append((tick % BAR, tick, n))
    return by


def _vote(groups, by_bar, key, merge, keep_half_conf):
    """Shared voting: for each group, positions (and keys) present in most of its bars win."""
    out, grouped = [], set()
    for g in groups:
        k = len(g)
        votes = defaultdict(list)
        for m in g:
            for pos, tick, n in by_bar.get(m, []):
                votes[(pos, key(n))].append((tick, n))
        pattern = []
        for (pos, kk), hits in votes.items():
            conf = np.mean([h[1][-1] for h in hits]) if len(hits[0][1]) > 3 else 0.5
            if 2 * len(hits) > k or (2 * len(hits) == k and conf >= keep_half_conf):
                pattern.append((pos, kk, hits))
        pattern = merge(pattern)
        for m in g:
            grouped.add(m)
            for pos, rec in pattern:
                out.append((m * BAR + pos, rec))
    return out, grouped


def consistent_fretted(notes, beatmap, groups, keep_half_conf=0.6):
    """notes: [(t, lanes, sustain_s, hopo, conf)] -> same format, repeats charted alike.
    beatmap must have its grid fitted (BeatMap.fit_grid) for these notes."""
    by_bar = _bars(notes, beatmap)

    def merge(pattern):
        # one note per position: the lane set with most votes (weighted by confidence)
        best = {}
        for pos, _, hits in pattern:
            for tick, n in hits:
                w = n[4] if len(n) > 4 else 0.5
                best.setdefault(pos, Counter())[n[1]] += w
        out = []
        for pos, c in best.items():
            lanes = c.most_common(1)[0][0]
            hits = [h for p, _, hs in pattern if p == pos for h in hs]
            lens = [beatmap.tick(n[0] + n[2]) - tick if n[2] > 0 else 0 for tick, n in hits]
            hopos = [bool(n[3]) for _, n in hits if n[3] is not None]
            conf = float(np.mean([n[4] if len(n) > 4 else 0.5 for _, n in hits]))
            out.append((pos, (lanes, int(np.median(lens)),
                              (sum(hopos) * 2 > len(hopos)) if hopos else None, conf)))
        return out

    # vote on positions first (any lanes), then pick lanes per position
    placed, grouped = _vote(groups, by_bar, key=lambda n: 0, merge=merge, keep_half_conf=keep_half_conf)
    out = [n for b, ns in by_bar.items() if b not in grouped for _, _, n in ns]
    for tick, (lanes, lt, hopo, conf) in placed:
        t = beatmap.time(tick)
        out.append((t, lanes, beatmap.time(tick + lt) - t if lt else 0.0, hopo, conf))
    return sorted(out, key=lambda n: n[0])


def consistent_drums(hits, beatmap, groups, keep_half_conf=0.6):
    """hits: [(t, lane, is_tom, conf)] -> same format, repeats charted alike."""
    by_bar = _bars(hits, beatmap)

    def merge(pattern):
        out = []
        for pos, lane, hs in pattern:
            toms = [bool(n[2]) for _, n in hs]
            conf = float(np.mean([n[3] if len(n) > 3 else 0.5 for _, n in hs]))
            out.append((pos, (lane, sum(toms) * 2 > len(toms), conf)))
        return out

    placed, grouped = _vote(groups, by_bar, key=lambda n: n[1], merge=merge, keep_half_conf=keep_half_conf)
    out = [n for b, ns in by_bar.items() if b not in grouped for _, _, n in ns]
    out += [(beatmap.time(tick), lane, tom, conf) for tick, (lane, tom, conf) in placed]
    return sorted(out, key=lambda n: (n[0], n[1]))


def make_consistent(kind, notes, feats, beatmap, thr=None):
    """Pipeline entry: fit the grid, find repeated bars from the features, vote them alike."""
    if not notes:
        return notes
    beatmap.fit_grid([n[0] for n in notes])
    n_bars = int(beatmap.raw_tick(len(feats) / features.FPS) // BAR) + 1
    groups = repeat_groups(bar_fingerprints(feats, beatmap, n_bars), thr or REPEAT_THR[kind])
    if kind == "drums":
        return consistent_drums(notes, beatmap, groups)
    return consistent_fretted(notes, beatmap, groups)
