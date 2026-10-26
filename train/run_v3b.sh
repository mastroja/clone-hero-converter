#!/bin/bash
# Run 2 of the v3 training plan (~7 h): guitar and bass also learn up/same/down fret moves and
# the decoder re-places frets along the melody; drums are retrained so converted five-lane
# charts teach kick/snare/timing but not cymbal vs tom. Needs run_v3a.sh's stems and features.
# GUITAR_SRC picks the guitar stem (ft = htdemucs_ft "other", s6 = htdemucs_6s "guitar"):
#   GUITAR_SRC=ft bash train/run_v3b.sh      (progress: data/v3/run.log)
# Pause: bash train/pause.sh   Resume: GUITAR_SRC=ft bash train/resume.sh b
cd "$(dirname "$0")/.."
source train/run_lib.sh
GUITAR_SRC=${GUITAR_SRC:-ft}
keep_awake "v3 part 2 done"
log "v3 part 2 start (guitar stems: $GUITAR_SRC)"
train() {  # name kind src [extra]
  step "train_$1" "$PY -m train.train $2 --data v3 --dm-src $3 --steps $4 --hidden 192 --dm-frac 0.6 $5 --out models/v3/$1.pt" \
       data/v3/train_$1.log
}
train guitar_rel guitar_hr $GUITAR_SRC 25000 --rel
train bass_rel bass_hr ft 25000 --rel
train drums_ct drums ft 20000
for m in guitar_rel:guitar bass_rel:bass; do
  step "calibrate_${m%:*}" "$PY -m train.calibrate ${m#*:} models/v3/${m%:*}.pt" data/v3/calibrate_${m%:*}.log
done
for m in guitar_rel:guitar bass_rel:bass drums_ct:drums; do
  step "compare_${m%:*}" "$PY -m train.compare ${m#*:} models/${m#*:}.pt models/v3/${m%:*}.pt" data/v3/compare_${m%:*}.log
  cat data/v3/compare_${m%:*}.log | grep -v Warning >> $LOG
done
log "v3 part 2 done"
