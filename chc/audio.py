"""Audio helpers: bundled ffmpeg, conversion to wav, wav reading."""
import subprocess
import wave
from pathlib import Path

import numpy as np

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW)
    if r.returncode:
        raise RuntimeError(r.stderr.decode("utf-8", "replace")[-400:])


def to_wav(src: Path, dest: Path) -> Path:
    """Decode any audio file to 44.1 kHz stereo 16-bit wav."""
    _run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
          "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(dest)])
    return dest


def to_ogg(wav: Path, dest: Path) -> Path:
    _run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(wav),
          "-c:a", "libvorbis", "-q:a", "6", str(dest)])
    return dest


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return (mono float32 samples, sample rate) from a 16-bit wav."""
    with wave.open(str(path), "rb") as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    """data: float array shaped (channels, samples)."""
    pcm = (np.clip(data, -1, 1).T * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[0])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()
