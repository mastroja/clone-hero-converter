"""Drop-in pipeline: audio file -> stems -> rough chart -> Clone Hero song folder."""
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import analysis, audio, chart

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
STEM_TO_CH = {"other": "guitar", "bass": "bass", "drums": "drums", "vocals": "vocals"}
STEPS = ["Reading your song", "Splitting it into instruments", "Finding the beat",
         "Writing the notes", "Saving the song folder"]
SEP = re.compile(r'[<>:"/\\|?*]')


class Cancelled(Exception):
    pass


@dataclass
class Result:
    song_dir: Path
    preview_dir: Path
    stems: dict = field(default_factory=dict)   # label -> preview wav path
    counts: dict = field(default_factory=dict)  # part -> expert note count
    bpm: float = 0.0
    seconds: float = 0.0


def parse_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if " - " in stem:
        a, t = stem.split(" - ", 1)
        return a.strip(), t.strip()
    return "", stem.replace("_", " ").strip()


def _separate(wav: Path, out: Path, model: str, on_progress, stop: threading.Event):
    proc = subprocess.Popen([sys.executable, "-m", "chc.worker", str(wav), str(out), model],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=Path(__file__).resolve().parent.parent, creationflags=NO_WINDOW)
    tail = []
    for line in proc.stdout:
        if stop.is_set():
            proc.terminate()
            proc.wait()
            raise Cancelled()
        if line.startswith("PROGRESS"):
            on_progress(float(line.split()[1]))
        else:
            tail.append(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError("Splitting failed:\n" + "\n".join(tail[-6:]))


def make_song(src: Path, artist: str, title: str, songs_dir: Path, charter: str = "",
              model: str = "htdemucs_ft", on_step=lambda i: None,
              on_progress=lambda frac: None, stop: threading.Event | None = None) -> Result:
    """Run everything. on_step(i) marks STEPS[i] as started; on_progress is 0..1 overall."""
    stop = stop or threading.Event()

    def check():
        if stop.is_set():
            raise Cancelled()

    name = SEP.sub("", f"{artist} - {title}" if artist else title).strip() or "Untitled"
    song_dir = songs_dir / name
    preview_dir = Path(tempfile.mkdtemp(prefix="chc_preview_"))
    work = Path(tempfile.mkdtemp(prefix="chc_work_"))
    try:
        on_step(0); on_progress(0.02)
        wav = audio.to_wav(src, work / "input.wav")
        length = audio.duration(wav)
        check()

        on_step(1)
        _separate(wav, preview_dir, model, lambda f: on_progress(0.05 + 0.65 * f), stop)
        check()

        on_step(2); on_progress(0.72)
        sr = 44100
        drums_x, sr = audio.read_wav(preview_dir / "drums.wav")
        drum_timed, env = analysis.drum_hits(drums_x, sr)
        bpm, offset = analysis.estimate_tempo(env, sr)
        check()

        on_step(3); on_progress(0.82)
        other_x, _ = audio.read_wav(preview_dir / "other.wav")
        pitched = analysis.pitched_notes(other_x, sr, 80, 1000)
        gtimed = [(t, l) for (t, _), l in zip(pitched, analysis.quantile_lanes([p for _, p in pitched]))]
        check()
        on_progress(0.9)
        bass_x, _ = audio.read_wav(preview_dir / "bass.wav")
        bp = analysis.pitched_notes(bass_x, sr, 35, 300, sensitivity=1.1, min_gap=0.12)
        btimed = [(t, l) for (t, _), l in zip(bp, analysis.quantile_lanes([p for _, p in bp]))]
        check()

        on_step(4); on_progress(0.93)
        song_dir.mkdir(parents=True, exist_ok=True)
        counts = chart.write_mid(song_dir / "notes.mid", bpm, offset,
                                 guitar=gtimed, bass=btimed, drums=drum_timed)
        chart.write_ini(song_dir / "song.ini", title, artist, charter, int(length * 1000), counts,
                        preview_ms=int(min(30, max(0, length - 40)) * 1000))
        for demucs_name, ch_name in STEM_TO_CH.items():
            audio.to_ogg(preview_dir / f"{demucs_name}.wav", song_dir / f"{ch_name}.ogg")
            check()
        on_progress(1.0)
        stems = {"Guitar & everything else": preview_dir / "other.wav",
                 "Bass": preview_dir / "bass.wav", "Drums": preview_dir / "drums.wav",
                 "Vocals": preview_dir / "vocals.wav"}
        return Result(song_dir, preview_dir, stems, counts, bpm, length)
    except BaseException:
        shutil.rmtree(preview_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
