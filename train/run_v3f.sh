#!/bin/bash
# Run 6 of the v3 plan (~6.5 h): fast runs and solos, where guitar misses the most notes (21% in
# fast passages vs 7% elsewhere, 26% inside solos). Two guitars, each tuned, compared with the
# app's guitar and diagnosed (the app's guitar is diagnosed again first, on the same songs):
#   focus: fast runs and solos weigh 3x in the loss and a quarter of windows sit in a solo
#          (same outputs as the app's guitar, so it could be switched in as it is)
#   fast:  the same, plus a lane-free note-start output for note times and the recurrent layer
#          at twice the time resolution
#   bash train/run_v3f.sh      (progress: data/v3/run.log)
# Pause: bash train/pause.sh   Resume: bash train/resume.sh f
cd "$(dirname "$0")/.."
source train/run_lib.sh
keep_awake "v3 part 6 done"
log "v3 part 6 start"
mkdir -p models/v3/app_diag && cp models/guitar.pt models/v3/app_diag/guitar.pt
step diagnose_guitar_app "$PY -m train.diagnose models/v3/app_diag" data/v3/diagnose_guitar_app.log
grep -v Warning data/v3/diagnose_guitar_app.log >> $LOG
for v in focus fast; do
  extra="--focus 3 --solo-frac 0.25"
  [ $v = fast ] && extra="$extra --onset --down 2"
  step train_guitar_$v "$PY -m train.train guitar_hr --data v3 --dm-src ft --steps 25000 --hidden 192 --dm-frac 0.6 --rel --size $extra --out models/v3/guitar_$v.pt" \
       data/v3/train_guitar_$v.log
  step calibrate_guitar_$v "$PY -m train.calibrate guitar models/v3/guitar_$v.pt" data/v3/calibrate_guitar_$v.log
  step compare_guitar_$v "$PY -m train.compare guitar models/guitar.pt models/v3/guitar_$v.pt" data/v3/compare_guitar_$v.log
  cat data/v3/compare_guitar_$v.log | grep -v Warning >> $LOG
  mkdir -p models/v3/${v}_diag && cp models/v3/guitar_$v.pt models/v3/${v}_diag/guitar.pt
  step diagnose_guitar_$v "$PY -m train.diagnose models/v3/${v}_diag" data/v3/diagnose_guitar_$v.log
  grep -v Warning data/v3/diagnose_guitar_$v.log >> $LOG
done
log "v3 part 6 done"
