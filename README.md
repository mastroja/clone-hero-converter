# clone-hero-converter

Automates the tedious first pass of charting: audio -> stems -> rough Clone Hero MIDI.

## Setup
Demucs needs the CUDA build of torch (a plain `pip install demucs` gives CPU-only):

    python -m pip uninstall torch torchvision torchaudio -y
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    python -m pip install demucs mido

Optional: put `ffmpeg` on PATH so stems are written as .ogg instead of .wav.

## 1. Split stems into song folders
    python separate.py "D:\inbox" -o "G:\Songs\Clone Hero" --charter you

Name inputs `Artist - Title.mp3`. Output per song: `guitar.ogg` (Demucs "other"),
`bass.ogg`, `drums.ogg`, `vocals.ogg`, and a templated `song.ini`.

## 2. Convert Ableton/Basic Pitch MIDI to Clone Hero MIDI
    python midi_to_ch.py guitar.mid "<song folder>\notes.mid" -i guitar
    python midi_to_ch.py drums.mid  "<song folder>\notes.mid" -i drums

Guitar notes are made monophonic, thinned (`--min-gap`), and spread over 5 lanes
(`--lane-mode quantile|contour|single`). Drums map General MIDI pitches
(kick 36, snare 38, hats 42, toms 41-50, cymbals 49/51/57) to pro-drum lanes with tom markers.
Tempo/time-signature events are copied from the source MIDI.
One instrument per run; to combine instruments, load each in Moonscraper and copy over.

## MIDI spec used (verified against TheNathannator/GuitarGame_ChartFormats)
- Track names: `PART GUITAR`, `PART BASS`, `PART DRUMS`; first track is the tempo map, plus `EVENTS`.
- Guitar/bass Expert: 96 green, 97 red, 98 yellow, 99 blue, 100 orange, 95 open (needs `[ENHANCED_OPENS]`),
  101 force HOPO, 102 force strum. Hard starts at 84, Medium 72, Easy 60.
  Star power 116, solo 103, tap 104.
- Drums Expert: 96 kick, 97 red, 98 yellow, 99 blue, 100 green (4-lane) / orange (5-lane), 101 5-lane green.
  Tom markers: 110 yellow, 111 blue, 112 green (unmarked = cymbal).
