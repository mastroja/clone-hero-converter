#!/bin/bash
# Run 2 of the v3 training plan: guitar and bass also learn up/same/down fret moves, and the
# decoder re-places frets along the melody (~6 h). Needs run_v3a.sh's stems and features.
# Set GUITAR_SRC to the guitar stem that won run 1 (ft or s6):
#   GUITAR_SRC=s6 bash train/run_v3b.sh      (progress: data/v3/run.log)
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
LOG=data/v3/run.log
GUITAR_SRC=${GUITAR_SRC:-ft}
log() { echo "$1 $(date)" >> $LOG; }
.venv/Scripts/pythonw.exe train/keep_awake.py "v3 part 2 done" $LOG &
log "v3 part 2 start (guitar stems: $GUITAR_SRC)"
train() {  # name kind src
  $PY -m train.train $2 --data v3 --dm-src $3 --steps 25000 --hidden 192 --dm-frac 0.6 --rel \
      --out models/v3/$1.pt > data/v3/train_$1.log 2>&1
  log "$1 trained: $(tail -1 data/v3/train_$1.log)"
}
train guitar_rel guitar_hr $GUITAR_SRC
train bass_rel bass_hr ft
for m in guitar_rel:guitar bass_rel:bass; do
  $PY -m train.calibrate ${m#*:} models/v3/${m%:*}.pt 2>&1 | grep -v "^  thr" >> $LOG
  echo "== ${m%:*} vs current app model" >> $LOG
  $PY -m train.compare ${m#*:} models/${m#*:}.pt models/v3/${m%:*}.pt >> $LOG 2>&1
done
log "v3 part 2 done"
