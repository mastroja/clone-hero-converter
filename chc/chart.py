"""Build a Clone Hero notes.mid and song.ini from timed note lists."""
from pathlib import Path

import mido
import numpy as np

TPQ = 480
GRID = TPQ // 4  # notes snap to 16ths
DIFF_BASE = {"expert": 96, "hard": 84, "medium": 72, "easy": 60}
# (min ticks between guitar notes, number of lanes used)
GUITAR_RULES = {"expert": (GRID, 5), "hard": (180, 5), "medium": (240, 4), "easy": (480, 3)}
# per drum lane: min ticks between hits (None = drop on this difficulty)
DRUM_RULES = {
    "expert": {"K": 0, "S": 0, "H": 0, "C": 0},
    "hard": {"K": 180, "S": 180, "H": 240, "C": 0},
    "medium": {"K": 240, "S": 480, "H": 480, "C": 960},
    "easy": {"K": 480, "S": 480, "H": None, "C": None},
}
DRUM_LANE = {"K": 0, "S": 1, "H": 2, "C": 4}  # offset from the difficulty's kick note


class TempoMap:
    """Constant tempo with a lead-in beat that absorbs the first-beat offset."""

    def __init__(self, bpm: float, offset: float, snap: bool = True):
        self.spb = 60.0 / bpm
        self.lead = offset + self.spb  # first real beat (tick TPQ) falls at this time
        self.grid = GRID if snap else 1  # snap=False keeps exact audio timing (1 tick ~ 1 ms)

    def tick(self, t: float) -> int:
        g = self.grid
        if t < self.lead:
            return int(round(t / self.lead * TPQ / g)) * g
        return TPQ + int(round((t - self.lead) / self.spb * TPQ / g)) * g


def grid_fit(times, bpm: float, offset: float, div: int = 4) -> float:
    """Mean distance of times from the nearest 1/div-beat gridline, as a fraction of the
    grid step: 0 = perfectly on grid, 0.25 = what random timing gives."""
    if len(times) == 0:
        return 0.25
    step = 60.0 / bpm / div
    r = ((np.asarray(times) - offset) / step) % 1.0
    return float(np.minimum(r, 1 - r).mean())


def _thin(notes, min_gap):
    out, last = [], -10 ** 9
    for tick, lane in notes:
        if tick - last >= min_gap:
            out.append((tick, lane))
            last = tick
    return out


def guitar_notes(timed, tm: TempoMap, difficulty: str):
    """timed: [(seconds, lane 0-4)] -> [(tick, lane)] reduced for the difficulty."""
    gap, lanes = GUITAR_RULES[difficulty]
    seen, notes = set(), []
    for t, lane in sorted(timed):
        tick = tm.tick(t)
        if tick in seen:
            continue
        seen.add(tick)
        notes.append((tick, lane * lanes // 5))
    return _thin(notes, gap)


def drum_notes(timed, tm: TempoMap, difficulty: str):
    """timed: [(seconds, 'K'|'S'|'H'|'C')] -> [(tick, lane_offset)]."""
    rules, last, out, seen = DRUM_RULES[difficulty], {}, [], set()
    for t, code in sorted(timed):
        gap = rules[code]
        tick = tm.tick(t)
        if gap is None or (tick, code) in seen:
            continue
        if tick - last.get(code, -10 ** 9) < gap:
            continue
        seen.add((tick, code))
        last[code] = tick
        out.append((tick, DRUM_LANE[code]))
    return out


def _track(name, events):
    tr = mido.MidiTrack([mido.MetaMessage("track_name", name=name, time=0)])
    last = 0
    order = {"note_off": 0, "note_on": 1}
    for tick, typ, note in sorted(events, key=lambda e: (e[0], order[e[1]])):
        tr.append(mido.Message(typ, note=note, velocity=100 if typ == "note_on" else 0,
                               time=tick - last))
        last = tick
    tr.append(mido.MetaMessage("end_of_track", time=0))
    return tr


def _events(notes_by_diff, drums=False):
    ev = []
    for diff, notes in notes_by_diff.items():
        for tick, lane in notes:
            n = DIFF_BASE[diff] + lane
            ev += [(tick, "note_on", n), (tick + 1, "note_off", n)]
    return ev


def write_mid(path: Path, bpm: float, offset: float, guitar=None, bass=None, drums=None,
              snap: bool = True):
    """Each part is a timed note list or None. Returns {part: expert note count}."""
    tm = TempoMap(bpm, offset, snap)
    mid = mido.MidiFile(type=1, ticks_per_beat=TPQ)
    tempo = mido.MidiTrack([
        mido.MetaMessage("track_name", name="", time=0),
        mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
        mido.MetaMessage("set_tempo", tempo=int(round(tm.lead * 1e6)), time=0),
        mido.MetaMessage("set_tempo", tempo=int(round(tm.spb * 1e6)), time=TPQ),
        mido.MetaMessage("end_of_track", time=0)])
    mid.tracks += [tempo, mido.MidiTrack([mido.MetaMessage("track_name", name="EVENTS", time=0),
                                          mido.MetaMessage("end_of_track", time=0)])]
    counts = {}
    for name, timed, fn in (("PART GUITAR", guitar, guitar_notes), ("PART BASS", bass, guitar_notes),
                            ("PART DRUMS", drums, drum_notes)):
        if not timed:
            continue
        by_diff = {d: fn(timed, tm, d) for d in DIFF_BASE}
        mid.tracks.append(_track(name, _events(by_diff)))
        counts[name] = len(by_diff["expert"])
    mid.save(path)
    return counts


def write_ini(path: Path, title: str, artist: str, charter: str, length_ms: int,
              parts: dict, preview_ms: int = 0):
    lines = ["[song]", f"name = {title}", f"artist = {artist}", "album =", "genre =", "year =",
             f"charter = {charter}", f"diff_guitar = {2 if 'PART GUITAR' in parts else -1}",
             f"diff_bass = {2 if 'PART BASS' in parts else -1}",
             f"diff_drums = {2 if 'PART DRUMS' in parts else -1}",
             "pro_drums = True", f"song_length = {length_ms}", "delay = 0",
             f"preview_start_time = {preview_ms}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
