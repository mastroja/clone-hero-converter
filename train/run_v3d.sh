#!/bin/bash
# Run 4 of the v3 plan (~3 h): guitar also learns how many frets each note has (--size), since
# wrong chord size was the biggest guitar error (41% of notes found at the right time). Tuned,
# compared with the app's guitar, then the error diagnosis is rerun on the new model.
#   bash train/run_v3d.sh      (progress: data/v3/run.log)
# Pause: bash train/pause.sh   Resume: bash train/resume.sh d
cd "$(dirname "$0")/.."
source train/run_lib.sh
keep_awake "v3 part 4 done"
log "v3 part 4 start"
step train_guitar_size "$PY -m train.train guitar_hr --data v3 --dm-src ft --steps 25000 --hidden 192 --dm-frac 0.6 --rel --size --out models/v3/guitar_size.pt" \
     data/v3/train_guitar_size.log
step calibrate_guitar_size "$PY -m train.calibrate guitar models/v3/guitar_size.pt" data/v3/calibrate_guitar_size.log
step compare_guitar_size "$PY -m train.compare guitar models/guitar.pt models/v3/guitar_size.pt" data/v3/compare_guitar_size.log
cat data/v3/compare_guitar_size.log | grep -v Warning >> $LOG
mkdir -p models/v3/size_diag && cp models/v3/guitar_size.pt models/v3/size_diag/guitar.pt
step diagnose_guitar_size "$PY -m train.diagnose models/v3/size_diag" data/v3/diagnose_guitar_size.log
grep -v Warning data/v3/diagnose_guitar_size.log >> $LOG
log "v3 part 4 done"
