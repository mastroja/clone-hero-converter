"""python train/keep_awake.py <marker> <log file> : keep Windows from idle-sleeping (the same
request a video player makes) until the log contains the marker, the run pauses, or 30 hours
pass."""
import ctypes
import sys
import time
from pathlib import Path

MARK, LOG = sys.argv[1], Path(sys.argv[2])
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
start = time.time()
while time.time() - start < 30 * 3600:
    if LOG.exists():
        text = LOG.read_text(errors="replace")
        lines = text.strip().splitlines()
        if MARK in text or (lines and lines[-1].startswith("paused")):
            break
    time.sleep(60)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
