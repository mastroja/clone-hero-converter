#!/bin/bash
# Quick end-to-end check of the v3 pipeline on a handful of songs (~10 min on the GPU):
# separate, features, train a few steps (with --rel), tune, compare with the app's models.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
OUT=data/v3/smoke
mkdir -p $OUT
IDS=$($PY - <<'E'
from train import data
idx = data.Layout("v3").index()
pick = {"train": [], "val": [], "test": []}
for s in idx:
    k = data.split_of(s)
    if "mix" in s["stems"] and all(i in s["notes"] for i in ("guitar", "bass", "drums")) and len(pick[k]) < (3 if k == "train" else 1):
        pick[k].append(data.song_id(s))
print(",".join(sum(pick.values(), [])))
E
)
echo "songs $IDS"
$PY -m train.separate_v3 "$IDS" 2>&1 | grep -v Warning
$PY -m train.feats_v3 all dm-only | tail -1
for k in "guitar_hr s6 --rel" "bass_hr ft --rel" "drums ft"; do
  set -- $k
  $PY -m train.train $1 --data v3 --dm-src $2 --steps 40 --hidden 64 --out $OUT/$1_$2.pt $3 2>&1 | grep -E "train songs|VAL|FINAL|Error|error"
done
$PY -m train.calibrate guitar $OUT/guitar_hr_s6.pt 2>&1 | grep -v "^  thr"
$PY -m train.compare guitar models/guitar.pt $OUT/guitar_hr_s6.pt val 2>&1 | grep -v Warning
$PY -m train.compare drums models/drums.pt $OUT/drums_ft.pt val 2>&1 | grep -v Warning
echo "smoke done"
