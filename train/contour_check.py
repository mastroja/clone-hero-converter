"""python -m train.contour_check [split] : does the chart move up and down where the real one does?

Exact frets are partly charter taste (two human charts of a song agree on ~60% of frets), but
the melody's shape (this note higher, lower or the same as the last) is what a player feels.
Scores, for guitar and bass on held-out songs:
  model      the app's current models, through the app's decoding
  random     the model's notes with random frets (floor)
  one fret   the model's notes all on one fret: only "same" moves agree (floor)
  humans     two human charts of the same song (songs charted twice in the library; ceiling)
"""
import sys

import numpy as np

from chc import ml
from chc import nn as cnn
from . import data, metrics
from .ceiling import best_offset, key
from .compare import load, songs


def pct(c):
    return c["direction"] / max(1, c["pairs"]), c["interval"] / max(1, c["pairs"])


def add(a, b):
    return {k: a.get(k, 0) + b[k] for k in b}


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    rng = np.random.default_rng(0)
    index = data.load_index()
    for inst in ("guitar", "bass"):
        ck, kind, model, layout = load(f"models/{inst}.pt")
        settings = {"thr": 0.35, "ratios": None, "hopo": 0.5, "sus": 0.5, "rel": 1.0, **ck.get("decode", {})}
        rows = list(songs(layout, inst, kind, split).values())
        tot = {"model": {}, "random": {}, "one fret": {}}
        for i, s in rows:
            p = cnn.predict(model, np.load(layout.dm(i, s, kind)), "cuda")
            got = [(n[0], n[1]) for n in ml.decode_fretted(inst, p, **settings)]
            gt = [(n[0], tuple(n[1])) for n in s["notes"][inst]]
            tot["model"] = add(tot["model"], metrics.contour(got, gt))
            rnd = [(t, (int(rng.integers(0, 5)),)) for t, _ in got]
            tot["random"] = add(tot["random"], metrics.contour(rnd, gt))
            tot["one fret"] = add(tot["one fret"], metrics.contour([(t, (0,)) for t, _ in got], gt))
        # humans: songs charted twice, lined up in time
        groups = {}
        for s in index:
            groups.setdefault(key(s["name"]), []).append(s)
        hum = {}
        for g in groups.values():
            if len(g) < 2 or inst not in g[0]["notes"] or inst not in g[1]["notes"]:
                continue
            a = [(n[0], tuple(n[1])) for n in g[0]["notes"][inst]]
            b = [(n[0], tuple(n[1])) for n in g[1]["notes"][inst]]
            off = best_offset([t for t, _ in a], [t for t, _ in b])
            c = metrics.contour([(t + off, l) for t, l in b], a)
            if c["pairs"] > 50:
                hum = add(hum, c)
        tot["humans"] = hum
        print(f"\n{inst} ({len(rows)} {split} songs; humans: songs charted twice)")
        print(f"  {'':10s} {'same direction':>15s} {'same jump size':>15s} {'note pairs':>11s}")
        for name in ("model", "random", "one fret", "humans"):
            d, iv = pct(tot[name])
            print(f"  {name:10s} {d:15.1%} {iv:15.1%} {tot[name].get('pairs', 0):11d}")


if __name__ == "__main__":
    main()
