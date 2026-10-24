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
from pathlib import Path

AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}
# Demucs stem -> Clone Hero audio file name
STEM_MAP = {"other": "guitar", "bass": "bass", "drums": "drums", "vocals": "vocals"}

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


def split_name(stem: str) -> tuple[str, str]:
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return artist.strip(), title.strip()
    return "Unknown", stem.strip()


def run_demucs(src: Path, tmp: Path, model: str, device: str) -> Path:
    cmd = [sys.executable, "-m", "demucs", "-n", model, "--device", device,
           "-o", str(tmp), str(src)]
    subprocess.run(cmd, check=True)
    return tmp / model / src.stem


def encode(wav: Path, dest: Path, ffmpeg: str | None) -> Path:
    if ffmpeg:
        out = dest.with_suffix(".ogg")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(wav),
                        "-c:a", "libvorbis", "-q:a", "6", str(out)], check=True)
        return out
    out = dest.with_suffix(".wav")
    shutil.copy2(wav, out)
    return out


def process(src: Path, out_root: Path, args, ffmpeg) -> None:
    artist, title = split_name(src.stem)
    song_dir = out_root / src.stem
    if song_dir.exists() and not args.force and (song_dir / "song.ini").exists():
        print(f"skip (exists): {src.name}")
        return
    song_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        stems = run_demucs(src, Path(t), args.model, args.device)
        for demucs_name, ch_name in STEM_MAP.items():
            wav = stems / f"{demucs_name}.wav"
            if wav.exists():
                encode(wav, song_dir / ch_name, ffmpeg)
    ini = song_dir / "song.ini"
    if not ini.exists() or args.force:
        ini.write_text(INI_TEMPLATE.format(title=title, artist=artist,
                                           charter=args.charter), encoding="utf-8")
    print(f"done: {song_dir}")


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

    files = [args.input] if args.input.is_file() else sorted(
        f for f in args.input.iterdir() if f.suffix.lower() in AUDIO_EXT)
    if not files:
        sys.exit("no audio files found")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found: stems will be kept as .wav (large)")
    args.out.mkdir(parents=True, exist_ok=True)
    for f in files:
        try:
            process(f, args.out, args, ffmpeg)
        except subprocess.CalledProcessError as e:
            print(f"FAILED {f.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
