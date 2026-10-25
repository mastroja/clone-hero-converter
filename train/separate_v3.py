"""python -m train.separate_v3 : run htdemucs_ft and htdemucs_6s on every song's full mix and
keep the stems the note models use (mono FLAC at the feature sample rate) under
data/v3/stems/<song id>/. Resumable. Keeping the stems means feature changes later only need
the fast feature step, not another night of separation."""
import subprocess
import time

import numpy as np
import torch

from chc import audio, features
from . import data
from .demucs_feats import full_mix, to_mono22

# (model, source) -> file name
KEEP = {("htdemucs_ft", "other"): "ft_other", ("htdemucs_ft", "bass"): "ft_bass",
        ("htdemucs_ft", "drums"): "ft_drums", ("htdemucs_6s", "guitar"): "s6_guitar"}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def save_flac(path, wave: np.ndarray):
    tmp = path.with_suffix(".part.flac")
    subprocess.run([audio.ffmpeg_exe(), "-v", "error", "-y", "-f", "f32le", "-ar", str(features.SR),
                    "-ac", "1", "-i", "-", "-sample_fmt", "s16", str(tmp)],
                   input=np.clip(wave, -1, 1).astype(np.float32).tobytes(), check=True,
                   creationflags=NO_WINDOW)
    tmp.replace(path)  # only complete files get the real name


def separate(model, mix: torch.Tensor, device):
    from demucs.apply import apply_model
    ref = mix.mean(0)
    x = (mix - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        src = apply_model(model, x[None].to(device), device=device, split=True, overlap=0.25)[0]
    return src * (ref.std() + 1e-8) + ref.mean()


def main():
    """python -m train.separate_v3 [max_songs | id,id,...]"""
    import sys
    from demucs.pretrained import get_model
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    limit = int(arg) if arg.isdigit() else 0
    only = set(arg.split(",")) if arg and not arg.isdigit() else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layout = data.Layout("v3")
    index = layout.index()
    models = {name: get_model(name).to(device).eval() for name in ("htdemucs_ft", "htdemucs_6s")}
    order = [s for s in index if "mix" in s["stems"] and (only is None or data.song_id(s) in only)]
    np.random.default_rng(0).shuffle(order)  # any prefix is a fair sample of the library
    t0, done = time.time(), 0
    for n, s in enumerate(order):
        if limit and done >= limit:
            break
        if data.paused():
            print("paused", flush=True)
            sys.exit(3)
        out = data.V3 / "stems" / data.song_id(s)
        if all((out / f"{f}.flac").exists() for f in KEEP.values()):
            continue
        try:
            mix = full_mix(s["stems"]["mix"])
            if mix.shape[1] < 44100 * 20:
                continue
            out.mkdir(parents=True, exist_ok=True)
            for name, model in models.items():
                need = [(src, f) for (m, src), f in KEEP.items() if m == name and not (out / f"{f}.flac").exists()]
                if not need:
                    continue
                stems = separate(model, mix, device)
                for src, f in need:
                    save_flac(out / f"{f}.flac", to_mono22(stems[model.sources.index(src)].cpu(), 44100))
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
