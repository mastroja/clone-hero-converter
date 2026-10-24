#!/usr/bin/env python
"""Batch audio files through Demucs (htdemucs_ft) into Clone Hero song folders.

Each input file "Artist - Title.mp3" becomes <out>/Artist - Title/ containing
guitar.ogg (Demucs "other"), bass.ogg, drums.ogg, vocals.ogg and a templated
song.ini. Stems are encoded to .ogg when ffmpeg is on PATH, otherwise kept .wav.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}
# Demucs stem -> Clone Hero audio file name
STEM_MAP = {"other": "guitar", "bass": "bass", "drums": "drums", "vocals": "vocals"}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

INI_TEMPLATE = """[song]
name = {title}
artist = {artist}
album =
genre =
year =
charter = {charter}
diff_guitar = -1
diff_bass = -1
diff_drums = -1
pro_drums = True
song_length = 0
delay = 0
preview_start_time = 0
"""


class Cancelled(Exception):
    pass


def split_name(stem: str) -> tuple[str, str]:
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return artist.strip(), title.strip()
    return "Unknown", stem.strip()


def collect_audio(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(f for f in path.iterdir() if f.suffix.lower() in AUDIO_EXT)


def run_streamed(cmd, log, stop: threading.Event | None):
    """Run cmd, forwarding output (split on \\r and \\n) to log; honours stop."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            creationflags=NO_WINDOW)
    buf = b""
    while True:
        if stop and stop.is_set():
            proc.terminate()
            proc.wait()
            raise Cancelled()
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch in (b"\r", b"\n"):
            if buf.strip():
                log(buf.decode("utf-8", "replace").strip(), progress=ch == b"\r")
            buf = b""
        else:
            buf += ch
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def encode(wav: Path, dest: Path, ffmpeg, log, stop) -> Path:
    if ffmpeg:
        out = dest.with_suffix(".ogg")
        run_streamed([ffmpeg, "-y", "-loglevel", "error", "-i", str(wav),
                      "-c:a", "libvorbis", "-q:a", "6", str(out)], log, stop)
        return out
    out = dest.with_suffix(".wav")
    shutil.copy2(wav, out)
    return out


def process(src: Path, out_root: Path, model, device, charter, force, ffmpeg, log, stop):
    artist, title = split_name(src.stem)
    song_dir = out_root / src.stem
    if song_dir.exists() and not force and (song_dir / "song.ini").exists():
        log(f"skip (exists): {src.name}")
        return
    song_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        run_streamed([sys.executable, "-m", "demucs", "-n", model, "--device", device,
                      "-o", t, str(src)], log, stop)
        stems = Path(t) / model / src.stem
        for demucs_name, ch_name in STEM_MAP.items():
            wav = stems / f"{demucs_name}.wav"
            if wav.exists():
                log(f"encoding {ch_name}")
                encode(wav, song_dir / ch_name, ffmpeg, log, stop)
    ini = song_dir / "song.ini"
    if not ini.exists() or force:
        ini.write_text(INI_TEMPLATE.format(title=title, artist=artist, charter=charter),
                       encoding="utf-8")
    log(f"done: {song_dir}")


def run_batch(files, out_root: Path, model="htdemucs_ft", device="cuda", charter="",
              force=False, log=lambda m, progress=False: print(m), stop: threading.Event | None = None,
              on_item=None) -> int:
    """Process files; returns number of failures. on_item(i, total, name) is optional."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log("ffmpeg not found: stems will be kept as .wav (large)")
    out_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for i, f in enumerate(files):
        if on_item:
            on_item(i, len(files), f.name)
        log(f"=== {f.name}")
        try:
            process(f, out_root, model, device, charter, force, ffmpeg, log, stop)
        except Cancelled:
            log("cancelled")
            return failures
        except subprocess.CalledProcessError as e:
            failures += 1
            log(f"FAILED {f.name}: {e}")
    if on_item:
        on_item(len(files), len(files), "")
    return failures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="audio file or folder of audio files")
    p.add_argument("-o", "--out", type=Path, required=True, help="Clone Hero songs folder")
    p.add_argument("--model", default="htdemucs_ft")
    p.add_argument("--device", default="cuda")
    p.add_argument("--charter", default="")
    p.add_argument("--force", action="store_true", help="redo songs that already exist")
    args = p.parse_args()
    files = collect_audio(args.input)
    if not files:
        sys.exit("no audio files found")
    run_batch(files, args.out, args.model, args.device, args.charter, args.force,
              log=lambda m, progress=False: print(m))


if __name__ == "__main__":
    main()
