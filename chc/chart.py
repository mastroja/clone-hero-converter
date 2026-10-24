"""Build a Clone Hero notes.mid and song.ini from timed note lists."""
from pathlib import Path

import mido
import numpy as np

TPQ = 480
GRID = TPQ // 4  # notes snap to 16ths
DIFF_BASE = {"expert": 96, "hard": 84, "medium": 72, "easy": 60}
# (min ticks between guitar notes, number of lanes used)
GUITAR_RULES = {"expert": (GRID, 5), "hard": (180, 5), "medium": (240, 4), "easy": (480, 3)}
# per drum lane (0 kick, 1 red, 2 yellow, 3 blue, 4 green): min ticks between hits, None = drop
DRUM_RULES = {
    "expert": [0, 0, 0, 0, 0],
    "hard": [180, 180, 240, 240, 240],
    "medium": [240, 480, 480, 480, 960],
    "easy": [480, 480, None, None, None],
}
TOM_MARK = {2: 110, 3: 111, 4: 112}


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


class BeatMap:
    """Variable tempo map that follows tracked beats, so charts line up beat by beat.

    Tempo is piecewise constant; each segment is the longest run of beats one tempo fits
    within `tol` seconds. A lead-in segment (before the first beat) lets bars start on a
    tracked downbeat.
    """

    def __init__(self, beats, downbeats=None, snap=True, tol=0.015):
        b = np.asarray(beats, float)
        self.snap = snap
        n = len(b)
        # lead-in: s beats between time 0 and the first tracked beat, so a downbeat lands on a bar line
        s = 1
        if downbeats is not None and len(downbeats):
            d = int(np.argmin(abs(b - downbeats[0])))
            s = (-d) % 4 or 4
        if not 0.2 <= b[0] / s <= 1.5:
            s = 1
        self.lead_beats, self.lead_len = s, max(b[0], 0.05)
        segs, i = [], 0
        while i < n - 1:
            j = i + 1
            while j + 1 < n:
                P = (b[j + 1] - b[i]) / (j + 1 - i)
                if np.max(np.abs(b[i:j + 2] - (b[i] + np.arange(j + 2 - i) * P))) > tol:
                    break
                j += 1
            segs.append((i, float(b[i]), float((b[j] - b[i]) / (j - i))))
            i = j
        if not segs:
            segs = [(0, float(b[0]), 0.5)]
        self.segs = segs
        self.seg_times = np.array([x[1] for x in segs])
        self.n_beats = n

    def raw_tick(self, t: float) -> float:
        s = self.lead_beats
        if t < self.segs[0][1]:
            return t / self.lead_len * s * TPQ
        k = max(0, int(np.searchsorted(self.seg_times, t, side="right")) - 1)
        i, t0, P = self.segs[k]
        return (i + s) * TPQ + (t - t0) / P * TPQ

    def tick(self, t: float) -> int:
        raw = self.raw_tick(t)
        if not self.snap:
            return int(round(raw))
        # snap to the nearest 16th or 12th-of-a-beat grid point if it is acoustically close
        k = max(0, int(np.searchsorted(self.seg_times, t, side="right")) - 1)
        spb = self.segs[k][2] if t >= self.segs[0][1] else self.lead_len / self.lead_beats
        best = None
        for g in (GRID, TPQ // 3):
            cand = int(round(raw / g)) * g
            err = abs(cand - raw) / TPQ * spb
            if best is None or err < best[0]:
                best = (err, cand)
        return best[1] if best[0] <= 0.025 else int(round(raw))

    def time(self, tick: float) -> float:
        """Inverse of raw_tick: seconds at a tick (for checking snapped notes)."""
        s = self.lead_beats
        if tick < s * TPQ:
            return tick / (s * TPQ) * self.lead_len
        k = max(0, int(np.searchsorted([(i + s) * TPQ for i, _, _ in self.segs], tick, side="right")) - 1)
        i, t0, P = self.segs[k]
        return t0 + (tick - (i + s) * TPQ) / TPQ * P

    def tempo_events(self):
        """[(tick, microseconds per quarter)]"""
        ev = [(0, int(round(self.lead_len / self.lead_beats * 1e6)))]
        for i, _, P in self.segs:
            ev.append(((i + self.lead_beats) * TPQ, int(round(P * 1e6))))
        return ev

    @property
    def bpm(self) -> float:
        Ps = np.array([P for _, _, P in self.segs])
        return round(float(60.0 / np.median(Ps)), 1)


def grid_fit(times, bpm: float, offset: float, div: int = 4) -> float:
    """Mean distance of times from the nearest 1/div-beat gridline, as a fraction of the
    grid step: 0 = perfectly on grid, 0.25 = what random timing gives."""
    if len(times) == 0:
        return 0.25
    step = 60.0 / bpm / div
    r = ((np.asarray(times) - offset) / step) % 1.0
    return float(np.minimum(r, 1 - r).mean())


HOPO_TICKS = TPQ // 3 + 1  # Clone Hero's automatic hammer-on distance for .mid charts
MAX_LANES = {"expert": 3, "hard": 2, "medium": 2, "easy": 1}
MIN_SUSTAIN_TICKS = TPQ // 2  # an 8th; anything shorter is written as a plain note
SUSTAIN_GAP = GRID // 2  # a sustain stops this far before the next note


def _thin(notes, min_gap):
    """notes: [(tick, ...)] sorted by tick."""
    out, last = [], -10 ** 9
    for n in notes:
        if n[0] - last >= min_gap:
            out.append(n)
            last = n[0]
    return out


def guitar_notes(timed, tm: TempoMap, difficulty: str):
    """timed: [(seconds, lane or lanes[, sustain_s, hammer_on])] -> [(tick, lane, length, force)]
    reduced for the difficulty. force is "hopo", "strum" or None (let the game decide)."""
    gap, nlanes = GUITAR_RULES[difficulty]
    by_tick = {}
    for n in timed:
        t, lanes = n[0], n[1]
        lanes = (lanes,) if isinstance(lanes, int) else tuple(lanes)
        length = n[2] if len(n) > 2 else 0.0
        hopo = n[3] if len(n) > 3 else None
        tick = tm.tick(t)
        cur = by_tick.setdefault(tick, [set(), 0, None])
        cur[0].update(l * nlanes // 5 for l in lanes)
        if length > 0:
            cur[1] = max(cur[1], tm.tick(t + length) - tick)
        if hopo is not None:
            cur[2] = hopo if cur[2] is None else cur[2] or hopo
    notes = []
    for tick in sorted(by_tick):
        lanes, length, hopo = by_tick[tick]
        notes.append((tick, tuple(sorted(lanes)[:MAX_LANES[difficulty]]), length, hopo))
    notes = _thin(notes, gap)
    out, prev = [], None
    for k, (tick, lanes, length, hopo) in enumerate(notes):
        nxt = notes[k + 1][0] if k + 1 < len(notes) else None
        if nxt is not None:
            length = min(length, nxt - tick - SUSTAIN_GAP)
        length = length if length >= MIN_SUSTAIN_TICKS else 1
        # hammer-ons only matter on expert/hard; the easier parts are spaced out and strummed
        force = None
        if hopo is not None and difficulty in ("expert", "hard") and prev is not None:
            natural = tick - prev[0] <= HOPO_TICKS and len(lanes) == 1 and lanes != prev[1]
            if hopo and not natural and lanes != prev[1]:
                force = "hopo"
            elif natural and not hopo:
                force = "strum"
        out += [(tick, lane, length, force) for lane in lanes]
        prev = (tick, lanes)
    return out


def drum_notes(timed, tm: TempoMap, difficulty: str):
    """timed: [(seconds, lane 0-4, is_tom)] -> [(tick, lane, is_tom)] reduced for the difficulty."""
    rules, last, out, seen = DRUM_RULES[difficulty], {}, [], set()
    for t, lane, tom in sorted(timed):
        gap = rules[lane]
        tick = tm.tick(t)
        if gap is None or (tick, lane) in seen:
            continue
        if tick - last.get(lane, -10 ** 9) < gap:
            continue
        seen.add((tick, lane))
        last[lane] = tick
        out.append((tick, lane, bool(tom)))
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
    """Guitar/bass notes are (tick, lane, length, force); drum notes are (tick, lane, is_tom)."""
    ev, marks = [], set()
    for diff, notes in notes_by_diff.items():
        for n in notes:
            tick, lane = n[0], n[1]
            note = DIFF_BASE[diff] + lane
            length = 1 if drums else n[2]
            ev += [(tick, "note_on", note), (tick + length, "note_off", note)]
            if drums and n[2] and lane in TOM_MARK:
                marks.add((tick, TOM_MARK[lane]))
            elif not drums and n[3]:
                marks.add((tick, DIFF_BASE[diff] + (5 if n[3] == "hopo" else 6)))
    for tick, mark in sorted(marks):
        ev += [(tick, "note_on", mark), (tick + 1, "note_off", mark)]
    return ev


def write_mid(path: Path, bpm: float, offset: float, guitar=None, bass=None, drums=None,
              snap: bool = True, beatmap: "BeatMap | None" = None):
    """Each part is a timed note list or None. Returns {part: expert note count}."""
    tm = beatmap if beatmap is not None else TempoMap(bpm, offset, snap)
    mid = mido.MidiFile(type=1, ticks_per_beat=TPQ)
    if beatmap is not None:
        tempo = mido.MidiTrack([mido.MetaMessage("track_name", name="", time=0),
                                mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0)])
        last = 0
        for tick, us in beatmap.tempo_events():
            tempo.append(mido.MetaMessage("set_tempo", tempo=us, time=tick - last))
            last = tick
        tempo.append(mido.MetaMessage("end_of_track", time=0))
    else:
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
        mid.tracks.append(_track(name, _events(by_diff, drums=fn is drum_notes)))
        counts[name] = len({n[0] for n in by_diff["expert"]})  # notes and chords, not lanes
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
