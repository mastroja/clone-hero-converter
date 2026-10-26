"""python -m train.fit_reduce : learn how charters reduce Expert to Hard / Medium / Easy (which
notes they keep) from the library's real charts, compare with the fixed rules the app used, and
save models/reduce.joblib for chc.reduce.

Scored on held-out (val + test) songs, starting from their real Expert charts: of the notes the
real Hard/Medium/Easy chart has, how many we keep (recall), how many we keep that it drops
(precision), and F1, plus how close the kept share is to the real one."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from chc import chart, reduce
from . import charts, data

INST = ("guitar", "bass", "drums")


def rule_keep(inst, level, notes, tpq):
    """The fixed rules (chart.DRUM_RULES / GUITAR_RULES) on Expert notes in ticks."""
    scale = tpq / chart.TPQ
    keep = np.zeros(len(notes), bool)
    if inst == "drums":
        gaps, last = chart.DRUM_RULES[level], {}
        for k, (t, lane, _) in enumerate(notes):
            g = gaps[lane]
            if g is None or t - last.get(lane, -1e18) < g * scale:
                continue
            keep[k], last[lane] = True, t
    else:
        gap, last = chart.GUITAR_RULES[level][0] * scale, -1e18
        for k, (t, _) in enumerate(notes):
            if t - last >= gap:
                keep[k], last = True, t
    return keep


def labels(inst, level, expert, reduced):
    if inst == "drums":
        have = {(t, l) for t, l, _ in reduced}
        return np.array([(t, l) in have for t, l, _ in expert])
    have = {t for t, _ in reduced}
    return np.array([t in have for t, _ in expert])


def main():
    index = data.Layout("v3").index()
    rows = {(i, l): [] for i in INST for l in reduce.LEVELS}
    for s in index:
        if s.get("five_lane"):
            continue
        try:
            lv = charts.parse_levels(Path(s["folder"]) / "notes.mid")
        except Exception:
            continue
        tpq, tm = lv["tpq"], lv["tempo"]
        spb = lambda t: tm.sec(t + tpq) - tm.sec(t)
        for inst in INST:
            levels = lv.get(inst)
            if not levels or len(levels["expert"]) < 50:
                continue
            ex = levels["expert"]
            X = reduce.drum_features(ex, tpq, spb) if inst == "drums" else reduce.fret_features(ex, tpq, spb)
            for level in reduce.LEVELS:
                if len(levels[level]) < 10:
                    continue  # level not charted
                y = labels(inst, level, ex, levels[level])
                rows[(inst, level)].append((data.split_of(s) == "train", X, y, rule_keep(inst, level, ex, tpq)))
    saved = {}
    for inst in INST:
        print(f"\n{inst}:")
        print(f"  {'':8s} {'rules F1':>9s} {'learned F1':>11s} {'real kept':>10s} {'rules kept':>11s} {'learned kept':>13s}")
        for level in reduce.LEVELS:
            r = rows[(inst, level)]
            if not r:
                continue
            tr = [x for x in r if x[0]]
            te = [x for x in r if not x[0]]
            Xtr, ytr = np.concatenate([x[1] for x in tr]), np.concatenate([x[2] for x in tr])
            model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(Xtr, ytr)
            # threshold: keep about as many notes as real charts do
            p = model.predict_proba(Xtr)[:, 1]
            thr = float(np.quantile(p, 1 - ytr.mean()))
            Xte, yte = np.concatenate([x[1] for x in te]), np.concatenate([x[2] for x in te])
            rk = np.concatenate([x[3] for x in te])
            lk = model.predict_proba(Xte)[:, 1] >= thr

            def f1(k):
                tp = np.sum(k & yte)
                return 2 * tp / max(1, k.sum() + yte.sum())
            print(f"  {level:8s} {f1(rk):9.3f} {f1(lk):11.3f} {yte.mean():10.0%} {rk.mean():11.0%} {lk.mean():13.0%}")
            # final model on all songs
            Xa, ya = np.concatenate([Xtr, Xte]), np.concatenate([ytr, yte])
            final = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(Xa, ya)
            pa = final.predict_proba(Xa)[:, 1]
            saved[(inst, level)] = {"model": final, "threshold": float(np.quantile(pa, 1 - ya.mean()))}
    joblib.dump(saved, reduce.MODEL_FILE)
    print(f"\nsaved {reduce.MODEL_FILE}")


if __name__ == "__main__":
    main()
