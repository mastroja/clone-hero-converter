"""python -m train.ceiling : how closely do two human charts of the same song agree?

Finds songs charted more than once in the library (same artist and title after removing
tags like "(Live)" or "(Harmonix)"), lines the two charts up in time, and scores one against
the other with the same metrics the models get. That is roughly the best a model could score."""
import re
from collections import defaultdict

import numpy as np

from . import data, metrics


def key(name: str) -> str:
    name = re.sub(r"[\(\[].*?[\)\]]", "", name.lower())
    return re.sub(r"[^a-z0-9]+", "", name)


def best_offset(a, b):
    """Shift (s) that lines up most onsets of chart b with chart a."""
    a, b = np.array(sorted(a)), np.array(sorted(b))
    if not len(a) or not len(b):
        return 0.0
    d = (a[None, :400] - b[:400, None]).ravel()
    d = d[abs(d) < 3.0]
    hist, edges = np.histogram(d, bins=np.arange(-3.0, 3.0001, 0.01))
    k = int(np.argmax(hist))
    near = d[(d >= edges[k] - 0.01) & (d <= edges[k + 1] + 0.01)]
    return float(np.median(near)) if len(near) else 0.0


def main():
    index = data.load_index()
    groups = defaultdict(list)
    for s in index:
        groups[key(s["name"])].append(s)
    pairs = [g[:2] for g in groups.values() if len(g) >= 2]
    print(f"{len(pairs)} songs charted twice")
    for inst in ("guitar", "bass", "drums"):
        rows, n = [], 0
        for a, b in pairs:
            if inst not in a["notes"] or inst not in b["notes"]:
                continue
            ea, eb = data.events(a, inst), data.events(b, inst)
            off = best_offset([t for t, _ in ea], [t for t, _ in eb])
            eb = [(t + off, c) for t, c in eb]
            sc = metrics.score(eb, ea)
            if sc["onset_tp"] < 0.3 * min(sc["n_pred"], sc["n_gt"]):
                continue  # not the same recording/arrangement
            rows.append(sc)
            n += 1
            print(f"  {inst}: {a['name']}  vs  {b['name']}")
        if rows:
            s = metrics.summarize(rows)
            print(f"{inst}: {n} pairs | notes at the same time F1 {s['onset_F1']:.3f} | exact frets/pads F1 "
                  f"{s['exact_F1']:.3f} | same frets when both have a note {s['lane_acc']:.3f}")


if __name__ == "__main__":
    main()
