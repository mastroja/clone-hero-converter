#!/bin/bash
# Run 1 of the v3 training plan (one GPU job at a time; about a day in total):
#   separate every song with htdemucs_ft + htdemucs_6s and keep the stems   (~12 h)
#   features: current + pitch-sharp (_hr), from original and Demucs stems  (~1 h)
#   train 20k steps: guitar_hr on ft "other", guitar_hr on 6s "guitar",
#                    bass_hr, drums incl. converted five-lane charts        (~9 h)
#   tune decoding, then compare each new model with the app's current one on test songs
# Start from the repo root:  bash train/run_v3a.sh   (progress: data/v3/run.log)
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
LOG=data/v3/run.log
mkdir -p data/v3 models/v3
log() { echo "$1 $(date)" >> $LOG; }
.venv/Scripts/pythonw.exe train/keep_awake.py "v3 part 1 done" $LOG &
log "v3 part 1 start"
[ -f data/v3/index.json ] || $PY -m train.build_index "G:/Clone Hero/songs/Quickplay" --v3 >> $LOG 2>&1
$PY -m train.separate_v3 > data/v3/separate.log 2>&1
log "separation done ($(grep -c FAILED data/v3/separate.log) failed)"
$PY -m train.feats_v3 > data/v3/feats.log 2>&1
log "features done"
train() {  # name kind src [extra]
  $PY -m train.train $2 --data v3 --dm-src $3 --steps 20000 --hidden 192 --dm-frac 0.6 \
      --out models/v3/$1.pt $4 > data/v3/train_$1.log 2>&1
  log "$1 trained: $(tail -1 data/v3/train_$1.log)"
}
train guitar_hr_ft guitar_hr ft
train guitar_hr_s6 guitar_hr s6
train bass_hr bass_hr ft
train drums drums ft
for m in guitar_hr_ft:guitar guitar_hr_s6:guitar bass_hr:bass; do
  $PY -m train.calibrate ${m#*:} models/v3/${m%:*}.pt 2>&1 | grep -v "^  thr" >> $LOG
done
for m in guitar_hr_ft:guitar guitar_hr_s6:guitar bass_hr:bass drums:drums; do
  echo "== ${m%:*} vs current app model" >> $LOG
  $PY -m train.compare ${m#*:} models/${m#*:}.pt models/v3/${m%:*}.pt >> $LOG 2>&1
done
log "v3 part 1 done"
