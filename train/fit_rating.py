"""python -m train.fit_rating [--generated] [--app-check] : learn song.ini difficulty tiers (0-6) per instrument
from the Quickplay library's own ratings and save models/rating.json for chc.rating.

Fits on train-split songs and reports on held-out (val + test) songs: exactly right, within one
tier, and the average miss, against always guessing the most common rating. --generated learns
from the charts our app models generate for each song instead of the real charts (what the app
will rate); --app-check rates our generated charts of held-out songs."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV

from chc import rating
from . import charts, data

INST = ("guitar", "bass", "drums")


def label(s, inst):
    v = charts.read_ini(Path(s["folder"]) / "song.ini").get(f"diff_{inst}", "")
    try:
        v = int(float(v))
    except ValueError:
        return None
    return v if 0 <= v <= 6 else None


def report(name, pred, y):
    pred = np.clip(np.round(pred), 0, 6)
    return (f"  {name:28s} exact {np.mean(pred == y):5.1%}   within one tier {np.mean(abs(pred - y) <= 1):5.1%}"
            f"   average miss {np.mean(abs(pred - y)):.2f}")


def generated_notes(index, inst):
    """{song folder: notes} as the app's current models chart each song."""
    from chc import ml
    from chc import nn as cnn
    model, _ = ml.load(inst)
    fkind = ml._meta[inst]["features"]
    out = {}
    for s in index:
        p = data.V3 / "feats" / f"{data.song_id(s)}_ft_{fkind}.npy"
        if inst not in s["notes"] or label(s, inst) is None or not p.exists():
            continue
        prob = cnn.predict(model, np.load(p), "cuda")
        out[s["folder"]] = ml.decode_drums(prob) if inst == "drums" else             ml.decode_fretted(inst, prob, **ml.decode_settings(inst))
    return out


def main():
    index = data.Layout("v3").index()
    out = {}
    generated = "--generated" in sys.argv
    for inst in INST:
        gen = generated_notes(index, inst) if generated else None
        rows = []
        for s in index:
            if inst not in s["notes"] or (gen is not None and s["folder"] not in gen):
                continue
            y = label(s, inst)
            f = rating.features(inst, gen[s["folder"]] if gen is not None else s["notes"][inst])
            if y is not None and f:
                rows.append((data.split_of(s) == "train", list(f), list(f.values()), y))
        names = rows[0][1]
        X = np.array([r[2] for r in rows], float)
        Y = np.array([r[3] for r in rows], float)
        tr = np.array([r[0] for r in rows])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = (X - mu) / sd
        ridge = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Z[tr], Y[tr])
        gbm = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                        subsample=0.8, random_state=0).fit(Z[tr], Y[tr])
        common = np.bincount(Y[tr].astype(int)).argmax()
        print(f"\n{inst}: {tr.sum()} songs to learn from, {(~tr).sum()} held out")
        print(report("always the most common (%d)" % common, np.full((~tr).sum(), common), Y[~tr]))
        print(report("linear model (saved)", ridge.predict(Z[~tr]), Y[~tr]))
        print(report("boosted trees (comparison)", gbm.predict(Z[~tr]), Y[~tr]))
        top = sorted(zip(names, ridge.coef_), key=lambda x: -abs(x[1]))[:5]
        print("  strongest signals:", ", ".join(f"{n} {c:+.2f}" for n, c in top))
        # final model on all labelled songs
        mu, sd = X.mean(0), X.std(0) + 1e-9
        final = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit((X - mu) / sd, Y)
        out[inst] = {"features": names, "mean": mu.tolist(), "std": sd.tolist(),
                     "coef": final.coef_.tolist(), "bias": float(final.intercept_)}
    rating.RATING_FILE.write_text(json.dumps(out, indent=1))
    print(f"\nsaved {rating.RATING_FILE}")
    if "--app-check" in sys.argv:
        app_check(index)


def app_check(index):
    """Rate our generated charts of held-out songs (decoded by the app's models)."""
    from chc import ml
    from chc import nn as cnn
    rating._model = None
    for inst in INST:
        model, _ = ml.load(inst)
        fkind = ml._meta[inst]["features"]
        preds, ys, real_preds = [], [], []
        for s in index:
            if data.split_of(s) == "train" or inst not in s["notes"]:
                continue
            y = label(s, inst)
            p = data.V3 / "feats" / f"{data.song_id(s)}_ft_{fkind}.npy"
            if y is None or not p.exists():
                continue
            prob = cnn.predict(model, np.load(p), "cuda")
            notes = ml.decode_drums(prob) if inst == "drums" else \
                ml.decode_fretted(inst, prob, **ml.decode_settings(inst))
            preds.append(rating.tier(inst, notes))
            real_preds.append(rating.tier(inst, s["notes"][inst]))
            ys.append(y)
        ys = np.array(ys, float)
        print(f"\n{inst}: {len(ys)} held-out songs, rated from our generated charts")
        print(report("from the real chart", np.array(real_preds, float), ys))
        print(report("from our generated chart", np.array(preds, float), ys))


if __name__ == "__main__":
    main()
