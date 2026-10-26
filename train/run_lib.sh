# Shared by the v3 run scripts: logging, pause handling, and skipping finished steps.
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8  # song names with non-Latin characters must not crash the logs
LOG=${LOG:-data/v3/run.log}
DONE=${DONE:-data/v3/done}
mkdir -p data/v3 models/v3 $DONE
log() { echo "$1 $(date)" >> $LOG; }

# keep_awake <marker>: start the keep-awake helper fully detached. A plain `&` child of bash
# does not survive on Windows, so go through PowerShell's Start-Process.
keep_awake() {
  powershell.exe -NoProfile -Command "Start-Process -WindowStyle Hidden -WorkingDirectory '$(pwd -W)' \
    -FilePath '.venv\\Scripts\\pythonw.exe' -ArgumentList 'train/keep_awake.py','\"$1\"','$LOG'"
}

# one run at a time (a run that is still finishing its pause holds the lock)
if [ -f data/v3/RUNNING ] && kill -0 "$(cat data/v3/RUNNING)" 2>/dev/null; then
  echo "A v3 run is still going (maybe finishing a pause); try again in a minute."
  exit 1
fi
echo $$ > data/v3/RUNNING
trap 'rm -f data/v3/RUNNING' EXIT

# step <name> <command> <log file>: run once; a finished step leaves data/v3/done/<name>
# and is skipped next time. Exit code 3 (or the pause file) means "paused": stop the run here.
step() {
  local name=$1 cmd=$2 out=$3
  [ -f $DONE/$name ] && return 0
  if [ -f data/v3/PAUSE ]; then log "paused before $name"; exit 0; fi
  log "$name start"
  $cmd >> $out 2>&1
  local rc=$?
  if [ $rc -eq 3 ] || [ -f data/v3/PAUSE ]; then log "paused during $name"; exit 0; fi
  if [ $rc -eq 0 ]; then touch $DONE/$name; log "$name done: $(grep -v Warning $out | tail -1)"
  else log "$name FAILED (exit $rc), see $out"; fi
}
