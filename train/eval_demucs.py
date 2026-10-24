"""python -m train.eval_demucs <kind> [split] [models/x/kind.pt] : score a trained model on Demucs-separated stems
(what the app really feeds it) versus the original isolated stems."""
import sys
from pathlib import Path

import numpy as np

from chc import audio, features, ml
from chc import nn as cnn
from . import data


def score(kind, probs_by_song, truths, thr):
    n_ch = cnn.n_lanes(kind)
    tp = fp = fn = 0
    atp = afp = afn = 0
    for p, tr in zip(probs_by_song, truths):
        pred = [(fr / features.FPS, c) for fr, c, _ in cnn.pick(p[:, :n_ch], thr)]
        a, b, c = data.f1(pred, tr, n_ch)
        tp, fp, fn = tp + a, fp + b, fn + c
        a, b, c = data.f1([(t, 0) for t, _ in pred], [(t, 0) for t, _ in tr], 1)
        atp, afp, afn = atp + a, afp + b, afn + c
    return data.prf(tp, fp, fn), data.prf(atp, afp, afn)


def main():
    kind = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    index = data.load_index()
    dm = Path("data/feats_dm")
    rows = [(i, s) for i, s in enumerate(index) if kind in s["notes"] and data.split_of(s) == split
            and (dm / f"{i:04d}_{kind}.npy").exists() and data.feat_path(i, kind).exists()]
    print(f"{len(rows)} {split} songs with Demucs features for {kind}")
    if len(sys.argv) > 3:  # a candidate model file instead of the one the app uses
        ml.MODEL_DIR = Path(sys.argv[3]).parent
        if Path(sys.argv[3]).stem != kind:
            raise SystemExit("model file must be named <kind>.pt")
    model, thr = ml.load(kind)
    clean, dem, truths = [], [], []
    for i, s in rows:
        clean.append(cnn.predict(model, np.load(data.feat_path(i, kind)), "cuda"))
        dem.append(cnn.predict(model, np.load(dm / f"{i:04d}_{kind}.npy"), "cuda"))
        truths.append(data.events(s, kind))
    for name, probs in (("original isolated stems", clean), ("Demucs stems", dem)):
        best = max(((score(kind, probs, truths, t), t) for t in (0.2, 0.3, 0.4, 0.5, 0.6)),
                   key=lambda x: x[0][0][2])
        (lane, anyl), t = best
        print(f"{name:26s} thr {t}: lane-aware P {lane[0]:.3f} R {lane[1]:.3f} F1 {lane[2]:.3f} | "
              f"any-lane F1 {anyl[2]:.3f}")


if __name__ == "__main__":
    main()
