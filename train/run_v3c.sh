#!/bin/bash
# Run 3 of the v3 plan (~5.5 h): the up/same/down guitar and bass models again, now also given
# the whole song's pitch profile (--context), since charters spread each song's range over the
# five frets and the models otherwise only see ~12 s at a time. Compared with the app's current
# models on melody shape, exact frets, timing, hammer-ons and sustains.
#   bash train/run_v3c.sh      (progress: data/v3/run.log)
# Pause: bash train/pause.sh   Resume: bash train/resume.sh c
cd "$(dirname "$0")/.."
source train/run_lib.sh
keep_awake "v3 part 3 done"
log "v3 part 3 start"
train() {  # name kind
  step "train_$1" "$PY -m train.train $2 --data v3 --dm-src ft --steps 25000 --hidden 192 --dm-frac 0.6 --rel --context --out models/v3/$1.pt" \
       data/v3/train_$1.log
}
train guitar_ctx guitar_hr
train bass_ctx bass_hr
for m in guitar_ctx:guitar bass_ctx:bass; do
  step "calibrate_${m%:*}" "$PY -m train.calibrate ${m#*:} models/v3/${m%:*}.pt" data/v3/calibrate_${m%:*}.log
  step "compare_${m%:*}" "$PY -m train.compare ${m#*:} models/${m#*:}.pt models/v3/${m%:*}.pt" data/v3/compare_${m%:*}.log
  cat data/v3/compare_${m%:*}.log | grep -v Warning >> $LOG
done
log "v3 part 3 done"
