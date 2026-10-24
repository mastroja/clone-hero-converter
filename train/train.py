"""python -m train.train <guitar|bass|drums> [--steps N] [--out models/x.pt] [--dm-frac 0.5]"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn as tnn

from chc import features
from chc import nn as cnn
from . import data

WIN = 1024  # frames per training window (~12 s); overridden by --win


class Bank:
    """Memory-mapped features plus lazily built targets for a set of songs."""

    def __init__(self, kind, items, path=data.feat_path):
        self.kind, self.items = kind, items
        self.n_ch = len(cnn.CHANNELS[kind])
        self.n_lanes = cnn.n_lanes(kind)
        self.feats, self.stats, self.evs, self.attrs = {}, {}, {}, {}
        for i, s in items:
            f = np.load(path(i, kind), mmap_mode="r")
            self.feats[i] = f
            sample = np.asarray(f[:: max(1, len(f) // 400)], dtype=np.float32)
            self.stats[i] = (float(sample.mean()), float(sample.std() + 1e-6))
            self.evs[i] = data.events(s, kind)
            if self.n_ch > self.n_lanes:
                self.attrs[i] = data.attributes(s, kind)
        # sample songs in proportion to their length
        self.w = np.array([len(self.feats[i]) for i, _ in items], float)
        self.w /= self.w.sum()

    def window(self, rng):
        i, s = self.items[rng.choice(len(self.items), p=self.w)]
        f = self.feats[i]
        a = int(rng.integers(0, max(1, len(f) - WIN)))
        x = np.asarray(f[a:a + WIN], dtype=np.float32)
        m, sd = self.stats[i]
        x = (x - m) / sd
        t0, t1 = a / features.FPS, (a + len(x)) / features.FPS
        ev = [(t - t0, c) for t, c in self.evs[i] if t0 - 0.1 <= t <= t1 + 0.1]
        y = np.zeros((len(x), self.n_ch), np.float32)
        y[:, :self.n_lanes] = data.targets(ev, len(x), self.n_lanes)
        if i in self.attrs:
            hopo, sus = self.attrs[i]
            y[:, self.n_lanes] = data.targets([(t - t0, 0) for t in hopo if t0 - 0.1 <= t <= t1 + 0.1],
                                              len(x), 1)[:, 0]
            for s0, s1 in sus:
                if s1 >= t0 and s0 <= t1:
                    lo = max(0, int(round((s0 - t0) * features.FPS)) + 2)
                    hi = min(len(x), int(round((s1 - t0) * features.FPS)))
                    y[lo:hi, self.n_lanes + 1] = 1.0
        if len(x) < WIN:
            x = np.pad(x, ((0, WIN - len(x)), (0, 0)))
            y = np.pad(y, ((0, WIN - len(y)), (0, 0)))
        return x, y


def attribute_scores(probs, songs, kind):
    """How well hammer-ons and sustains are called on the true notes: (hopo P/R/F1, sustain P/R/F1)."""
    h = [0, 0, 0]
    s = [0, 0, 0]
    nl = cnn.n_lanes(kind)
    for p, song in zip(probs, songs):
        for n in song["notes"][data.base(kind)]:
            if len(n) < 4:
                continue
            fr = int(round(n[0] * features.FPS))
            if not 0 <= fr < len(p):
                continue
            ph = p[max(0, fr - 2): fr + 3, nl].max() > 0.5
            ps = p[fr + 4: fr + 9, nl + 1].mean() > 0.5 if fr + 9 <= len(p) else False
            for acc, pred, true in ((h, ph, bool(n[3])), (s, ps, n[2] > 0)):
                if pred and true:
                    acc[0] += 1
                elif pred:
                    acc[1] += 1
                elif true:
                    acc[2] += 1
    return data.prf(*h), data.prf(*s)


def evaluate(model, items, kind, device, thresholds=None, grid=(0.2, 0.3, 0.4, 0.5, 0.6),
             path=data.feat_path):
    """Return (best lane-aware (P,R,F1), thr), any-lane (P,R,F1), attribute scores or None."""
    n_l = cnn.n_lanes(kind)
    probs, truths = [], []
    for i, s in items:
        probs.append(cnn.predict(model, np.load(path(i, kind)), device))
        truths.append(data.events(s, kind))
    best = None
    for thr in (grid if thresholds is None else [thresholds]):
        tp = fp = fn = 0
        for p, tr in zip(probs, truths):
            pred = [(fr / features.FPS, c) for fr, c, _ in cnn.pick(p[:, :n_l], thr)]
            a, b, c = data.f1(pred, tr, n_l)
            tp, fp, fn = tp + a, fp + b, fn + c
        res = (data.prf(tp, fp, fn), thr)
        if best is None or res[0][2] > best[0][2]:
            best = res
    # lane-agnostic score at the best threshold (is there a note here at all?)
    tp = fp = fn = 0
    for p, tr in zip(probs, truths):
        pred = [(fr / features.FPS, 0) for fr, c, _ in cnn.pick(p[:, :n_l], best[1])]
        a, b, c = data.f1(pred, [(t, 0) for t, _ in tr], 1)
        tp, fp, fn = tp + a, fp + b, fn + c
    attrs = None
    if model.n_out > n_l:
        attrs = attribute_scores(probs, [s for _, s in items], kind)
    return best, data.prf(tp, fp, fn), attrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(cnn.CHANNELS))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-songs", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--dm-frac", type=float, default=0.5,
                    help="share of training windows taken from Demucs-separated stems")
    args = ap.parse_args()
    global WIN
    WIN = args.win
    out = Path(args.out or f"models/{args.kind}.pt")
    out.parent.mkdir(exist_ok=True)
    device = "cuda"
    index = data.load_index()
    train_items = data.songs_for(args.kind, "train", index)
    if args.max_songs:
        train_items = train_items[:args.max_songs]
    dm_items = [(i, s) for i, s in train_items if data.dm_path(i, args.kind).exists()]
    val_dm = [(i, s) for i, s in data.songs_for(args.kind, "val", index) if data.dm_path(i, args.kind).exists()]
    # score on Demucs stems when we have them: that is what the app feeds the model
    val_items, val_path = (val_dm, data.dm_path) if val_dm else (data.songs_for(args.kind, "val", index), data.feat_path)
    print(f"{args.kind}: {len(train_items)} train ({len(dm_items)} with Demucs stems) / "
          f"{len(val_items)} val songs ({'Demucs' if val_dm else 'original'} stems)", flush=True)
    banks = [Bank(args.kind, train_items)]
    frac = 0.0
    if dm_items and args.dm_frac > 0:
        banks.append(Bank(args.kind, dm_items, data.dm_path))
        frac = args.dm_frac
    rng = np.random.default_rng(0)
    model = cnn.NoteNet(args.kind, args.hidden).to(device)
    n_l, n_out = cnn.n_lanes(args.kind), model.n_out
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.steps, pct_start=0.1)
    pw = torch.full((n_out,), 6.0, device=device)
    if n_out > n_l:
        pw[n_l], pw[n_l + 1] = 3.0, 1.0  # hammer-on flags are sparse; sustain spans are not
    lossf = tnn.BCEWithLogitsLoss(pos_weight=pw)
    t0 = time.time()
    run, best_f = 0.0, -1.0

    def save(thr):
        torch.save({"state": model.state_dict(), "kind": args.kind, "thr": thr, "hidden": args.hidden,
                    "n_out": n_out}, out)

    for step in range(1, args.steps + 1):
        model.train()
        xs, ys = zip(*(banks[int(len(banks) > 1 and rng.random() < frac)].window(rng)
                       for _ in range(args.batch)))
        x = torch.from_numpy(np.stack(xs)).to(device)
        y = torch.from_numpy(np.stack(ys)).to(device)
        x = x + 0.05 * torch.randn_like(x)  # light noise augmentation
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lossf(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        tnn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0:
            print(f"step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
        if step % 1500 == 0 or step == args.steps:
            ((p, r, f), thr), anyl, attrs = evaluate(model, val_items, args.kind, device, path=val_path)
            msg = f"  VAL thr {thr}: P {p:.3f} R {r:.3f} F1 {f:.3f} | any-lane F1 {anyl[2]:.3f}"
            if attrs:
                msg += f" | hopo F1 {attrs[0][2]:.3f} sustain F1 {attrs[1][2]:.3f}"
            print(msg, flush=True)
            if f > best_f:  # keep the best checkpoint, not just the last
                best_f = f
                save(thr)
    print(f"FINAL best val lane-aware F1 {best_f:.3f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
