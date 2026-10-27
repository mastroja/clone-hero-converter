#!/bin/bash
# Run 5 of the v3 plan (~12 h): separate every song again keeping full-quality stems (drums at
# 44.1 kHz; stereo guitar, keys and "other"; vocals), retrain drums on full-bandwidth audio
# (hi-hats and cymbals live above 11 kHz) and compare with the app's drums, including how often
# hi-hats and snares get mixed up; then train a stereo guitar model and compare with the app's.
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
# stereo guitar (~3.5 h): the middle and sides of the stereo guitar audio plus the keys stem as
# three input channels, so the model can tell a centred lead from hard-panned rhythm guitars,
# and guitar from keys; otherwise the same setup as the app's guitar (part 4) for a fair test
step features_guitar_st "$PY -m train.feats_v3 guitar_st" data/v3/feats_guitar_st.log
step train_guitar_st "$PY -m train.train guitar_st --data v3 --dm-src ft --steps 25000 --hidden 192 --dm-frac 0.6 --rel --size --out models/v3/guitar_st.pt" \
     data/v3/train_guitar_st.log
step calibrate_guitar_st "$PY -m train.calibrate guitar models/v3/guitar_st.pt" data/v3/calibrate_guitar_st.log
step compare_guitar_st "$PY -m train.compare guitar models/guitar.pt models/v3/guitar_st.pt" data/v3/compare_guitar_st.log
cat data/v3/compare_guitar_st.log | grep -v Warning >> $LOG
mkdir -p models/v3/st_diag && cp models/v3/guitar_st.pt models/v3/st_diag/guitar.pt && cp models/bass.pt models/v3/st_diag/bass.pt
step diagnose_guitar_st "$PY -m train.diagnose models/v3/st_diag" data/v3/diagnose_guitar_st.log
grep -v Warning data/v3/diagnose_guitar_st.log >> $LOG
log "v3 part 5 done"
