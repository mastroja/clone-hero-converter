"""Drop-in pipeline: audio file -> stems -> rough chart -> Clone Hero song folder."""
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import analysis, audio, chart, ml

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
        drums_x, sr = audio.read_wav(preview_dir / "drums.wav")
        beatmap = None
        if ml.available("mix"):
            mix_x, mix_sr = audio.read_wav(wav)
            b, d = ml.beat_times(mix_x, mix_sr)
            if len(b) >= 16:
                beatmap = chart.BeatMap(b, d)
        heur_drums, env = analysis.drum_hits(drums_x, sr)
        bpm, offset = analysis.estimate_tempo(env, sr)
        if beatmap is not None:
            bpm = beatmap.bpm
        check()

        on_step(3); on_progress(0.80)
        other_x, _ = audio.read_wav(preview_dir / "other.wav")
        bass_x, _ = audio.read_wav(preview_dir / "bass.wav")
        if ml.available("drums"):
            drum_timed = ml.drum_notes(drums_x, sr)
        else:
            lane_of = {"K": 0, "S": 1, "H": 2, "C": 4}
            drum_timed = [(t, lane_of[c], False) for t, c in heur_drums]
        check()
        on_progress(0.86)
        if ml.available("guitar"):
            gtimed = ml.fretted_notes("guitar", other_x, sr)
        else:
            att = analysis.guitar_attacks(other_x, sr)
            gt_ = [t for t, _ in att]
            gtimed = list(zip(gt_, analysis.local_lanes([h for _, h in att], gt_)))
        check()
        on_progress(0.90)
        if ml.available("bass"):
            btimed = ml.fretted_notes("bass", bass_x, sr)
        else:
            bp = analysis.pitched_notes(bass_x, sr, 35, 300, sensitivity=1.1, min_gap=0.12)
            bt_ = [t for t, _ in bp]
            btimed = list(zip(bt_, analysis.local_lanes([p for _, p in bp], bt_)))
        # Only snap to the beat grid when the song really sits on one; live or loose
        # recordings are charted at their exact audio times instead.
        fit = chart.grid_fit([g[0] for g in gtimed] + [d[0] for d in drum_timed], bpm, offset)
        snap = fit < 0.15 and beatmap is None
        check()

        on_step(4); on_progress(0.93)
        song_dir.mkdir(parents=True, exist_ok=True)
        counts = chart.write_mid(song_dir / "notes.mid", bpm, offset,
                                 guitar=gtimed, bass=btimed, drums=drum_timed, snap=snap,
                                 beatmap=beatmap)
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
