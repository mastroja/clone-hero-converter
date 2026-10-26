"""python -m train.separate_hq [max_songs | id,id,...] : separate every song again (htdemucs_ft and
htdemucs_6s) and keep higher-quality stems next to the v3 ones in data/v3/stems/<song id>/:

  ft_drums44.flac    drums, mono, 44.1 kHz (hi-hats and cymbals live above 11 kHz)
  ft_other_st.flac   htdemucs_ft "other", stereo, 22.05 kHz
  s6_guitar_st.flac  htdemucs_6s guitar, stereo, 22.05 kHz
  s6_piano_st.flac   htdemucs_6s piano/keys, stereo, 22.05 kHz
  ft_vocals.flac     vocals, mono, 22.05 kHz

The stereo stems let a guitar model hear what is panned where (doubled rhythm guitars are often
hard left/right, leads in the middle) and tell guitar from keys. Resumable; pauses between songs."""
import subprocess
import sys
import time

import numpy as np
import torch

from chc import audio, features
from . import data
from .demucs_feats import full_mix
from .separate_v3 import separate

# (model, source) -> (file name, channels, sample rate)
KEEP = {("htdemucs_ft", "drums"): ("ft_drums44", 1, 44100),
        ("htdemucs_ft", "other"): ("ft_other_st", 2, features.SR),
        ("htdemucs_ft", "vocals"): ("ft_vocals", 1, features.SR),
        ("htdemucs_6s", "guitar"): ("s6_guitar_st", 2, features.SR),
        ("htdemucs_6s", "piano"): ("s6_piano_st", 2, features.SR)}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def save_flac(path, wave: torch.Tensor, channels: int, sr: int):
    """wave: (2, n) at 44.1 kHz from Demucs -> FLAC with the given channels and rate."""
    import torchaudio
    w = wave.mean(0, keepdim=True) if channels == 1 else wave
    if sr != 44100:
        w = torchaudio.functional.resample(w, 44100, sr)
    pcm = np.clip(w.T.cpu().numpy(), -1, 1).astype(np.float32)  # interleaved
    tmp = path.with_suffix(".part.flac")
    subprocess.run([audio.ffmpeg_exe(), "-v", "error", "-y", "-f", "f32le", "-ar", str(sr),
                    "-ac", str(channels), "-i", "-", "-sample_fmt", "s16", str(tmp)],
                   input=pcm.tobytes(), check=True, creationflags=NO_WINDOW)
    tmp.replace(path)


def main():
    from demucs.pretrained import get_model
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    limit = int(arg) if arg.isdigit() else 0
    only = set(arg.split(",")) if arg and not arg.isdigit() else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = data.Layout("v3").index()
    models = {name: get_model(name).to(device).eval() for name in ("htdemucs_ft", "htdemucs_6s")}
    order = [s for s in index if "mix" in s["stems"] and (only is None or data.song_id(s) in only)]
    np.random.default_rng(0).shuffle(order)
    t0, done = time.time(), 0
    for n, s in enumerate(order):
        if limit and done >= limit:
            break
        if data.paused():
            print("paused", flush=True)
            sys.exit(3)
        out = data.V3 / "stems" / data.song_id(s)
        if all((out / f"{f}.flac").exists() for f, _, _ in KEEP.values()):
            continue
        try:
            mix = full_mix(s["stems"]["mix"])
            if mix.shape[1] < 44100 * 20:
                continue
            out.mkdir(parents=True, exist_ok=True)
            for name, model in models.items():
                need = [(src, spec) for (m, src), spec in KEEP.items()
                        if m == name and not (out / f"{spec[0]}.flac").exists()]
                if not need:
                    continue
                stems = separate(model, mix, device)
                for src, (f, ch, sr) in need:
                    save_flac(out / f"{f}.flac", stems[model.sources.index(src)].cpu(), ch, sr)
                del stems
                if device == "cuda":
                    torch.cuda.empty_cache()
            done += 1
            print(f"{n} {len(order)} {s['name']} ({(time.time() - t0) / done:.0f}s/song)", flush=True)
        except Exception as e:  # keep going on bad files
            print("FAILED", s["name"], e, flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
