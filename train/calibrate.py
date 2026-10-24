"""python -m train.calibrate <guitar|bass> <model.pt> : tune how the app turns probabilities into
notes (onset threshold, how sure extra chord lanes must be, hammer-on and sustain thresholds) on
validation songs separated by Demucs, and save the settings into the model file."""
import itertools
import sys
from collections import Counter

import numpy as np
import torch

from chc import features, ml
from chc import nn as cnn
from . import data, metrics


def chord_mix(notes):
    c = Counter(min(len(n[1]), 3) for n in notes)
    n = max(1, sum(c.values()))
    return " ".join(f"{k}:{c[k] / n:.0%}" for k in (1, 2, 3))


def main():
    kind, path = sys.argv[1], sys.argv[2]
    ck = torch.load(path, map_location="cuda", weights_only=False)
    model = cnn.NoteNet(kind, ck.get("hidden", 128), ck["state"]["out.weight"].shape[0]).cuda()
    model.load_state_dict(ck["state"])
    index = data.load_index()
    rows = [(i, s) for i, s in enumerate(index) if kind in s["notes"] and data.split_of(s) == "val"
            and data.dm_path(i, kind).exists()]
    probs = [cnn.predict(model, np.load(data.dm_path(i, kind)), "cuda") for i, _ in rows]
    truths = [data.events(s, kind) for _, s in rows]
    gt_notes = [n for _, s in rows for n in s["notes"][kind]]
    print(f"{len(rows)} val songs; real charts: {chord_mix(gt_notes)}")

    def chart_score(thr, ratios):
        res, notes = [], []
        for p, tr in zip(probs, truths):
            got = ml.decode_fretted(kind, p, thr, ratios)
            notes += got
            res.append(metrics.score([(t, l) for t, lanes, *_ in got for l in lanes], tr))
        return metrics.summarize(res), notes

    def chord_share(notes, size=2):
        return sum(len(n[1]) >= size for n in notes) / max(1, len(notes))

    # Accuracy alone prefers no chords at all (each guessed extra fret is often wrong), but a
    # chart without chords feels wrong too: require at least half the real charts' share of
    # chords, and of triple chords.
    need = 0.5 * chord_share(gt_notes)
    need3 = 0.5 * chord_share(gt_notes, 3) if ml.MAX_CHORD[kind] >= 3 else 0.0
    best = fallback = None
    hi = ml.MAX_CHORD[kind] > 2
    for thr, r1, r2 in itertools.product((0.25, 0.3, 0.35, 0.4, 0.45), (0.7, 0.8, 0.85, 0.9, 0.95, 1.01),
                                         (0.9, 0.95, 1.01) if hi else (1.01,)):
        if r2 < r1:
            continue
        s, notes = chart_score(thr, (r1, r2))
        ok = chord_share(notes) >= need and chord_share(notes, 3) >= need3
        print(f"  thr {thr} ratios ({r1}, {r2}): exact-frets F1 {s['exact_F1']:.3f} onset F1 "
              f"{s['onset_F1']:.3f} chords {chord_mix(notes)}{'' if ok else '  (too few chords)'}", flush=True)
        # exact_F1: the note is there at the right time AND has exactly the right frets
        if ok and (best is None or s["exact_F1"] > best[0]["exact_F1"]):
            best = (s, thr, (r1, r2), notes)
        if fallback is None or s["exact_F1"] > fallback[0]["exact_F1"]:
            fallback = (s, thr, (r1, r2), notes)
    best = best or fallback  # nothing met the chord rules: take the most accurate
    s, thr, ratios, notes = best
    print(f"best onset thr {thr} chord ratios {ratios}: onset F1 {s['onset_F1']:.3f} "
          f"exact-frets F1 {s['exact_F1']:.3f} fret accuracy {s['lane_acc']:.3f}; chords {chord_mix(notes)}")
    decode = {"thr": thr, "ratios": list(ratios), "hopo": 0.5, "sus": 0.5}
    if model.n_out >= 7:
        # hammer-on / sustain thresholds judged on the real notes' positions
        nl = cnn.n_lanes(kind)
        for ch, key, true_of in ((nl, "hopo", lambda n: bool(n[3])), (nl + 1, "sus", lambda n: n[2] > 0)):
            scores = []
            for th in (0.3, 0.4, 0.5, 0.6, 0.7):
                tp = fp = fn = 0
                for p, (_, song) in zip(probs, rows):
                    for n in song["notes"][kind]:
                        fr = int(round(n[0] * features.FPS))
                        if fr + 9 > len(p):
                            continue
                        v = p[max(0, fr - 2): fr + 3, ch].max() if key == "hopo" else p[fr + 4: fr + 9, ch].mean()
                        pr, tr = v > th, true_of(n)
                        tp, fp, fn = tp + (pr and tr), fp + (pr and not tr), fn + (tr and not pr)
                scores.append((data.prf(tp, fp, fn)[2], th))
            f, th = max(scores)
            decode[key] = th
            print(f"{key}: threshold {th} F1 {f:.3f}")
    ck["decode"] = decode
    torch.save(ck, path)
    print("saved", decode, "->", path)


if __name__ == "__main__":
    main()
