"""python -m train.compare <guitar|bass|drums> <old.pt> <new.pt> [split]

Scores two models on the same held-out songs (test split by default) the way the app uses
them: each model gets its own Demucs stems and features and its own tuned decoding settings,
and the resulting chart notes are compared with the real charts."""
import sys
from collections import Counter

import numpy as np
import torch

from chc import features, ml
from chc import nn as cnn
from . import data, metrics


def load(path):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    kind = ck.get("kind")
    m = cnn.NoteNet(kind, ck.get("hidden", 128), ck["state"]["out.weight"].shape[0]).cuda()
    m.load_state_dict(ck["state"])
    m.eval()
    return ck, kind, m, data.Layout(ck.get("data", "v2"), ck.get("dm_src", "ft"))


def songs(layout, inst, kind, split):
    return {s["folder"]: (i, s) for i, s in enumerate(layout.index())
            if inst in s["notes"] and data.split_of(s) == split and layout.dm(i, s, kind).exists()}


def drum_channel(lane, tom):
    return 0 if lane == 0 else 1 if lane == 1 else 2 + (lane - 2) * 2 + int(tom)


def score(inst, ck, kind, model, layout, rows):
    if inst == "drums":
        tp = fp = fn = atp = afp = afn = 0
        for i, s in rows:
            p = cnn.predict(model, np.load(layout.dm(i, s, kind)), "cuda")
            got = ml.decode_drums(p, ck.get("thr"))
            pred = [(t, drum_channel(l, tom)) for t, l, tom in got]
            truth = data.events(s, inst)
            a, b, c = data.f1(pred, truth, 8)
            tp, fp, fn = tp + a, fp + b, fn + c
            a, b, c = data.f1([(t, 0) for t, _ in pred], [(t, 0) for t, _ in truth], 1)
            atp, afp, afn = atp + a, afp + b, afn + c
        return {"pad-and-type F1": data.prf(tp, fp, fn)[2], "any-hit F1": data.prf(atp, afp, afn)[2]}
    settings = {"thr": 0.35, "ratios": None, "hopo": 0.5, "sus": 0.5, "rel": 1.0, **ck.get("decode", {})}
    res, notes = [], []
    h = [0, 0, 0]
    su = [0, 0, 0]
    for i, s in rows:
        p = cnn.predict(model, np.load(layout.dm(i, s, kind)), "cuda")
        got = ml.decode_fretted(inst, p, **settings)
        notes += got
        truth = data.events(s, inst)
        res.append(metrics.score([(t, l) for t, lanes, *_ in got for l in lanes], truth))
        # hammer-ons and sustains on notes found at the right time
        times = np.array([g[0] for g in got])
        for n in s["notes"][inst]:
            if len(n) < 4 or not len(times):
                continue
            j = int(np.argmin(abs(times - n[0])))
            if abs(times[j] - n[0]) > metrics.TOL:
                continue
            for acc, pred, true in ((h, bool(got[j][3]), bool(n[3])), (su, got[j][2] > 0, n[2] > 0)):
                if pred and true:
                    acc[0] += 1
                elif pred:
                    acc[1] += 1
                elif true:
                    acc[2] += 1
    sm = metrics.summarize(res)
    c = Counter(min(len(n[1]), 3) for n in notes)
    tot = max(1, sum(c.values()))
    return {"notes at the right time F1": sm["onset_F1"], "exact frets F1": sm["exact_F1"],
            "fret accuracy": sm["lane_acc"], "hammer-on F1": data.prf(*h)[2],
            "sustain F1": data.prf(*su)[2],
            "chords 1/2/3": " / ".join(f"{c[k] / tot:.0%}" for k in (1, 2, 3))}


def main():
    inst, old_p, new_p = sys.argv[1:4]
    split = sys.argv[4] if len(sys.argv) > 4 else "test"
    old, new = load(old_p), load(new_p)
    a = songs(old[3], inst, old[1], split)
    b = songs(new[3], inst, new[1], split)
    common = sorted(set(a) & set(b))
    print(f"{inst}: {len(common)} {split} songs in both ({len(a)} old, {len(b)} new available)")
    so = score(inst, *old, [a[f] for f in common])
    sn = score(inst, *new, [b[f] for f in common])
    print(f"{'':28s} {'old':>12s} {'new':>12s}")
    for k in so:
        fo = f"{so[k]:.3f}" if isinstance(so[k], float) else so[k]
        fn = f"{sn[k]:.3f}" if isinstance(sn[k], float) else sn[k]
        print(f"{k:28s} {fo:>12s} {fn:>12s}")
    if len(b) > len(common):  # new layouts can add songs (e.g. converted five-lane drums)
        sb = score(inst, *new, list(b.values()))
        print(f"new model on all {len(b)} of its {split} songs: " +
              ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in sb.items()))


if __name__ == "__main__":
    main()
