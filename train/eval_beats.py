"""python -m train.eval_beats [split] : beat F-measure (+-70 ms) of network + DP tracker."""
import sys
import numpy as np
import torch
from chc import beats, features, nn as cnn
from . import data


def fmeasure(est, ref, tol=0.07):
    est, ref = np.asarray(est), np.asarray(ref)
    tp = sum(1 for r in ref if len(est) and np.min(abs(est - r)) <= tol)
    p = tp / max(1, len(est)); r = tp / max(1, len(ref))
    return 2 * p * r / max(1e-9, p + r)


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    idx = data.load_index()
    ck = torch.load("models/mix.pt", map_location="cuda", weights_only=False)
    m = cnn.NoteNet("mix").cuda(); m.load_state_dict(ck["state"]); m.eval()
    fs, ds, raw = [], [], []
    for i, s in data.songs_for("mix", split, idx):
        p = cnn.predict(m, np.load(data.feat_path(i, "mix")), "cuda")
        b, d = beats.track(p[:, 0], p[:, 1])
        fs.append(fmeasure(b, s["beats"])); ds.append(fmeasure(d, s["downbeats"], 0.1))
        pk = [f / features.FPS for f, _, _ in cnn.pick(p[:, :1], 0.4)]
        raw.append(fmeasure(pk, s["beats"]))
    print(f"{split}: {len(fs)} songs | beat F (DP) {np.mean(fs):.3f} | beat F (raw peaks) {np.mean(raw):.3f} | "
          f"downbeat F {np.mean(ds):.3f} | songs with beat F>0.8: {np.mean(np.array(fs) > 0.8):.2f}")


if __name__ == "__main__":
    main()
