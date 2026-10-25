"""python -m train.feats_v3 [kinds] : cache note-model features for the v3 layout from
(a) each song's own isolated stems ('clean') and (b) the Demucs stems kept by separate_v3
('ft', 's6'). Resumable; runs in minutes per hundred songs, so feature ideas are cheap to try."""
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from chc import features
from chc import nn as cnn
from . import data
from .make_features import load_stem

KINDS = ["guitar", "guitar_hr", "bass", "bass_hr", "drums"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Demucs stem file -> (source tag, instrument)
DM = {"ft_other": ("ft", "guitar"), "ft_bass": ("ft", "bass"), "ft_drums": ("ft", "drums"),
      "s6_guitar": ("s6", "guitar")}


def jobs_for(song, kinds):
    sid = data.song_id(song)
    out = data.V3 / "feats"
    for kind in kinds:
        inst = cnn.instrument(kind)
        if inst not in song["notes"]:
            continue
        if inst in song["stems"]:
            yield song["stems"][inst], out / f"{sid}_clean_{kind}.npy", kind
        for stem, (src, i2) in DM.items():
            p = data.V3 / "stems" / sid / f"{stem}.flac"
            if i2 == inst and p.exists():
                yield [str(p)], out / f"{sid}_{src}_{kind}.npy", kind


def main():
    """python -m train.feats_v3 [kinds|all] [only songs with Demucs stems: 'dm-only']"""
    kinds = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else KINDS
    (data.V3 / "feats").mkdir(parents=True, exist_ok=True)
    index = data.Layout("v3").index()
    if "dm-only" in sys.argv:  # quick tests: just the songs separate_v3 has done
        index = [s for s in index if (data.V3 / "stems" / data.song_id(s)).exists()]
    # one decode per audio file, all its feature kinds at once
    todo = {}
    for s in index:
        for paths, dest, kind in jobs_for(s, kinds):
            if not dest.exists():
                todo.setdefault(tuple(paths), []).append((dest, kind))
    items = list(todo.items())
    print(len(items), "audio files,", sum(len(v) for v in todo.values()), "feature files to build", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        for k in range(0, len(items), 24):  # bounded batches keep memory flat
            batch = items[k:k + 24]
            for (paths, outs), wave in zip(batch, ex.map(lambda it: load_stem(list(it[0])), batch)):
                if len(wave) < features.SR:
                    continue
                for dest, kind in outs:
                    data.save_npy(dest, features.compute(wave, kind, DEVICE))
                    done += 1
            print(done, "feature files", flush=True)
            if data.paused():
                print("paused", flush=True)
                sys.exit(3)
    print("done", done)


if __name__ == "__main__":
    main()
