#!/bin/bash
# Pause the v3 training run at its next safe point (between songs, or within ~100 training
# steps after saving its state). Nothing is lost; resume with train/resume.sh.
cd "$(dirname "$0")/.."
mkdir -p data/v3
touch data/v3/PAUSE
echo "pause requested $(date)" >> data/v3/run.log
echo "Pausing: the run stops at its next safe point (usually within a minute or two)."
