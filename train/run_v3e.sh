#!/bin/bash
# Run 5 of the v3 plan (~9 h): separate every song again keeping full-quality stems (drums at
# 44.1 kHz; stereo guitar, keys and "other"; vocals) for this and the next guitar experiment,
# then retrain drums on full-bandwidth audio (hi-hats and cymbals live above 11 kHz) and compare
# with the app's drums, including how often hi-hats and snares get mixed up.
#   bash train/run_v3e.sh      (progress: data/v3/run.log)
# Pause: bash train/pause.sh   Resume: bash train/resume.sh e
cd "$(dirname "$0")/.."
source train/run_lib.sh
keep_awake "v3 part 5 done"
log "v3 part 5 start"
step separate_hq "$PY -m train.separate_hq" data/v3/separate_hq.log
step features_hq "$PY -m train.feats_v3 drums_hq" data/v3/feats_hq.log
step train_drums_hq "$PY -m train.train drums_hq --data v3 --dm-src ft --steps 20000 --hidden 192 --dm-frac 0.6 --out models/v3/drums_hq.pt" \
     data/v3/train_drums_hq.log
step compare_drums_hq "$PY -m train.compare drums models/drums.pt models/v3/drums_hq.pt" data/v3/compare_drums_hq.log
cat data/v3/compare_drums_hq.log | grep -v Warning >> $LOG
log "v3 part 5 done"
