#!/usr/bin/env python
"""Convert a generic MIDI file (e.g. from Ableton audio-to-MIDI) into a Clone
Hero notes.mid with correct track names and note numbers.

Note numbers follow the community spec (TheNathannator/GuitarGame_ChartFormats):
  Guitar/Bass  Expert: 96 G, 97 R, 98 Y, 99 B, 100 O   (Hard +84, Med +72, Easy +60)
  Drums        Expert: 96 kick, 97 R, 98 Y, 99 B, 100 G (4-lane pro), toms marked
               with 110 Y / 111 B / 112 G; cymbals are the default.
Tempo/time-signature events are copied from the source so the chart lines up
with the audio; fix the BPM markers in Moonscraper if the source tempo is off.
"""
import argparse
from pathlib import Path

import mido

GUITAR_BASE = 96  # expert green
DRUM_KICK, DRUM_R, DRUM_Y, DRUM_B, DRUM_G = 96, 97, 98, 99, 100
TOM_MARK = {DRUM_Y: 110, DRUM_B: 111, DRUM_G: 112}
TRACK_NAMES = {"guitar": "PART GUITAR", "bass": "PART BASS", "drums": "PART DRUMS"}

# General MIDI drum note -> (expert note, is_tom)
GM_DRUMS = {
    35: (DRUM_KICK, False), 36: (DRUM_KICK, False),
    37: (DRUM_R, False), 38: (DRUM_R, False), 39: (DRUM_R, False), 40: (DRUM_R, False),
    42: (DRUM_Y, False), 44: (DRUM_Y, False), 46: (DRUM_Y, False),
    48: (DRUM_Y, True), 50: (DRUM_Y, True),
    45: (DRUM_B, True), 47: (DRUM_B, True),
    41: (DRUM_G, True), 43: (DRUM_G, True),
    49: (DRUM_G, False), 51: (DRUM_G, False), 52: (DRUM_G, False),
    53: (DRUM_B, False), 55: (DRUM_G, False), 57: (DRUM_G, False), 59: (DRUM_B, False),
}


def read_notes(mid: mido.MidiFile, track_sel: str | None):
    """Return [(start_tick, end_tick, pitch, velocity)] from the chosen track."""
    tracks = []
    for i, tr in enumerate(mid.tracks):
        t, open_, notes = 0, {}, []
        for m in tr:
            t += m.time
            if m.type == "note_on" and m.velocity > 0:
                open_.setdefault(m.note, (t, m.velocity))
            elif m.type in ("note_off", "note_on") and m.note in open_:
                s, v = open_.pop(m.note)
                notes.append((s, t, m.note, v))
        if notes:
            tracks.append((i, tr.name, notes))
    if not tracks:
        raise SystemExit("no notes found in input MIDI")
    if track_sel is None:
        return max(tracks, key=lambda x: len(x[2]))[2]
    for i, name, notes in tracks:
        if track_sel == str(i) or track_sel.lower() == name.lower():
            return notes
    raise SystemExit("track not found; available: " +
                     ", ".join(f"{i}:{n!r}" for i, n, _ in tracks))


def monophonize(notes, tol):
    """Merge notes starting within tol ticks into one, keeping the loudest."""
    out = []
    for n in sorted(notes):
        if out and n[0] - out[-1][0] <= tol:
            if n[3] > out[-1][3]:
                out[-1] = n
        else:
            out.append(n)
    return out


def quantile_lanes(pitches, lanes=5):
    s = sorted(pitches)
    cuts = [s[int(len(s) * k / lanes)] for k in range(1, lanes)]
    return [sum(p >= c for c in cuts) for p in pitches]


def map_guitar(notes, tpq, mode, min_gap):
    notes = monophonize(notes, tpq // 16)
    # drop notes that follow the previous one too closely (noise)
    kept = []
    for n in notes:
        if kept and n[0] - kept[-1][0] < min_gap:
            continue
        kept.append(n)
    pitches = [n[2] for n in kept]
    if mode == "quantile":
        lanes = quantile_lanes(pitches) if kept else []
    elif mode == "single":
        lanes = [0] * len(kept)
    else:  # contour: up/down relative to previous note, wrapping within 0-4
        lanes, cur = [], 2
        for i, p in enumerate(pitches):
            if i:
                cur += (p > pitches[i - 1]) - (p < pitches[i - 1])
                cur = max(0, min(4, cur))
            lanes.append(cur)
    return [(s, e, lane) for (s, e, _, _), lane in zip(kept, lanes)]


def build_track(name, events):
    """events: [(tick, msg_type, note, velocity)] -> MidiTrack with deltas."""
    tr = mido.MidiTrack([mido.MetaMessage("track_name", name=name, time=0)])
    last = 0
    order = {"note_off": 0, "note_on": 1}
    for tick, typ, note, vel in sorted(events, key=lambda e: (e[0], order[e[1]])):
        tr.append(mido.Message(typ, note=note, velocity=vel, channel=0, time=tick - last))
        last = tick
    tr.append(mido.MetaMessage("end_of_track", time=0))
    return tr


def guitar_events(notes, tpq, args):
    mapped = map_guitar(notes, tpq, args.lane_mode, int(tpq * args.min_gap))
    ev = []
    sus_min = tpq // 2
    for i, (s, e, lane) in enumerate(mapped):
        nxt = mapped[i + 1][0] if i + 1 < len(mapped) else None
        end = s + 1
        if args.sustains and e - s >= sus_min:
            end = e if nxt is None else min(e, nxt - tpq // 8)
            end = max(end, s + 1)
        ev += [(s, "note_on", GUITAR_BASE + lane, 100), (end, "note_off", GUITAR_BASE + lane, 0)]
    return ev


def drum_events(notes, tpq):
    ev, seen = [], set()
    for s, e, p, v in sorted(notes):
        if p not in GM_DRUMS:
            continue
        note, tom = GM_DRUMS[p]
        if (s, note) in seen:
            continue
        seen.add((s, note))
        ev += [(s, "note_on", note, 100), (s + 1, "note_off", note, 0)]
        if tom:
            m = TOM_MARK[note]
            ev += [(s, "note_on", m, 100), (s + 1, "note_off", m, 0)]
    return ev


def list_tracks(path):
    """[(index, name, note_count)] for tracks that contain notes."""
    mid = mido.MidiFile(path)
    return [(i, tr.name, sum(m.type == "note_on" and m.velocity > 0 for m in tr))
            for i, tr in enumerate(mid.tracks)
            if any(m.type == "note_on" for m in tr)]


def convert(input, output=None, instrument="guitar", track=None,
            lane_mode="quantile", min_gap=0.125, sustains=False):
    """Write a Clone Hero MIDI; returns (output_path, note_count)."""
    args = argparse.Namespace(instrument=instrument, lane_mode=lane_mode,
                              min_gap=min_gap, sustains=sustains)
    input = Path(input)
    src = mido.MidiFile(input)
    tpq = src.ticks_per_beat
    notes = read_notes(src, track)
    events = drum_events(notes, tpq) if instrument == "drums" else guitar_events(notes, tpq, args)

    out = mido.MidiFile(type=1, ticks_per_beat=tpq)
    tempo = mido.MidiTrack()  # tempo map: copy meta events from the source, drop names
    for tr in src.tracks:
        t = 0
        for m in tr:
            t += m.time
            if m.type in ("set_tempo", "time_signature"):
                tempo.append((t, m.copy(time=0)))
    tempo.sort(key=lambda x: x[0])
    tm, last = mido.MidiTrack([mido.MetaMessage("track_name", name="", time=0)]), 0
    for t, m in tempo:
        tm.append(m.copy(time=t - last)); last = t
    tm.append(mido.MetaMessage("end_of_track", time=0))
    out.tracks += [tm, mido.MidiTrack([mido.MetaMessage("track_name", name="EVENTS", time=0),
                                       mido.MetaMessage("end_of_track", time=0)]),
                   build_track(TRACK_NAMES[instrument], events)]
    dest = Path(output) if output else input.with_name("notes.mid")
    out.save(dest)
    return dest, sum(e[1] == "note_on" for e in events)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path, nargs="?", help="default: notes.mid next to input")
    p.add_argument("-i", "--instrument", choices=TRACK_NAMES, default="guitar")
    p.add_argument("-t", "--track", help="source track index or name (default: busiest)")
    p.add_argument("--lane-mode", choices=["quantile", "contour", "single"], default="quantile")
    p.add_argument("--min-gap", type=float, default=0.125,
                   help="min beats between guitar notes (thins noisy transcriptions)")
    p.add_argument("--sustains", action="store_true", help="keep long notes as sustains")
    a = p.parse_args()
    dest, n = convert(a.input, a.output, a.instrument, a.track, a.lane_mode, a.min_gap, a.sustains)
    print(f"wrote {dest}: {n} note-ons on {TRACK_NAMES[a.instrument]}")


if __name__ == "__main__":
    main()
