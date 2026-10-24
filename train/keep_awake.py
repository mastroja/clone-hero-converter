"""python train/keep_awake.py <marker> <log file> : keep Windows from idle-sleeping (the same
request a video player makes) until the log contains the marker, or 30 hours pass."""
import ctypes
import sys
import time
from pathlib import Path

MARK, LOG = sys.argv[1], Path(sys.argv[2])
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
start = time.time()
while time.time() - start < 30 * 3600:
    if LOG.exists() and MARK in LOG.read_text(errors="replace"):
        break
    time.sleep(60)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
