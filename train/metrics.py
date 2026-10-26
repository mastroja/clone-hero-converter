"""Chart-level scoring: merge lanes into notes/chords, then compare onsets and lane sets."""
import numpy as np

TOL = 0.05


def merge(events, win=0.04):
    """[(t, ch)] -> [(t, frozenset(channels))] with near-simultaneous events grouped."""
    out = []
    for t, c in sorted(events):
        if out and t - out[-1][0] <= win:
            out[-1][1].add(c)
        else:
            out.append([t, {c}])
    return [(t, frozenset(s)) for t, s in out]


def score(pred_events, gt_events, tol=TOL):
    """Return counts: onset tp/fp/fn, exact (time + same lane set) tp, per-channel tp/fp/fn."""
    p, g = merge(pred_events), merge(gt_events)
    pt = np.array([t for t, _ in p])
    used = np.zeros(len(p), bool)
    onset_tp = exact_tp = 0
    for t, lanes in g:
        if len(p):
            j = np.argmin(np.where(used, 1e9, abs(pt - t)))
            if not used[j] and abs(pt[j] - t) <= tol:
                used[j] = True
                onset_tp += 1
                exact_tp += p[j][1] == lanes
    return {"onset_tp": onset_tp, "onset_fp": len(p) - onset_tp, "onset_fn": len(g) - onset_tp,
            "exact_tp": exact_tp, "n_pred": len(p), "n_gt": len(g)}


def match(pred_times, gt_times, tol=TOL):
    """One-to-one nearest matching: {pred index: gt index} for notes within tol seconds."""
    pt, gt = np.asarray(pred_times), np.asarray(gt_times)
    used, out = np.zeros(len(pt), bool), {}
    for j, t in enumerate(gt):
        if not len(pt):
            break
        k = int(np.argmin(np.where(used, 1e9, abs(pt - t))))
        if not used[k] and abs(pt[k] - t) <= tol:
            used[k] = True
            out[k] = j
    return out


def contour(pred, gt, tol=TOL):
    """Melody shape agreement between two charts. pred, gt: [(time, lanes)] sorted by time.

    For every two notes in a row that are found in both charts (and are also neighbours in the
    real one), compare how the lowest fret moves: up, same or down. Returns counts
    {"pairs", "direction" (moves the same way), "interval" (moves by the same number of frets)}.
    """
    m = match([p[0] for p in pred], [g[0] for g in gt], tol)
    sign = lambda d: (d > 0) - (d < 0)
    pairs = same_dir = same_int = 0
    for k in range(len(pred) - 1):
        if k in m and k + 1 in m and m[k + 1] == m[k] + 1:
            dp = min(pred[k + 1][1]) - min(pred[k][1])
            dg = min(gt[m[k + 1]][1]) - min(gt[m[k]][1])
            pairs += 1
            same_dir += sign(dp) == sign(dg)
            same_int += dp == dg
    return {"pairs": pairs, "direction": same_dir, "interval": same_int}


def summarize(rows):
    s = {k: sum(r[k] for r in rows) for k in rows[0]}
    P = s["onset_tp"] / max(1, s["n_pred"])
    R = s["onset_tp"] / max(1, s["n_gt"])
    F = 2 * P * R / max(1e-9, P + R)
    xp = s["exact_tp"] / max(1, s["n_pred"])
    xr = s["exact_tp"] / max(1, s["n_gt"])
    xf = 2 * xp * xr / max(1e-9, xp + xr)
    return {"onset_P": P, "onset_R": R, "onset_F1": F, "lane_acc": s["exact_tp"] / max(1, s["onset_tp"]),
            "exact_F1": xf}
