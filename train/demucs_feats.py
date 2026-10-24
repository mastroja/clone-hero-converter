"""python -m train.demucs_feats [split ...] : run Demucs on full mixes and cache note-model
features of the resulting stems (other->guitar, bass, drums), so models can be trained on the
same kind of audio the app produces. Resumable; wavs are never kept."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from chc import features
from . import data
from .make_features import decode

OUT = Path("data/feats_dm")
STEM_KIND = {"other": "guitar", "bass": "bass", "drums": "drums"}


def full_mix(paths) -> torch.Tensor:
    """Stereo 44.1 kHz sum of every stem file: the song as originally released."""
    from chc import audio
    mix = None
    for p in paths:
        r = subprocess.run([audio.ffmpeg_exe(), "-v", "error", "-i", p, "-ac", "2", "-ar", "44100",
                            "-f", "f32le", "-"], capture_output=True)
        w = np.frombuffer(r.stdout, np.float32).reshape(-1, 2).T
        if mix is None:
            mix = np.zeros_like(w)
        if w.shape[1] > mix.shape[1]:
            mix = np.pad(mix, ((0, 0), (0, w.shape[1] - mix.shape[1])))
        mix[:, :w.shape[1]] += w
    return torch.from_numpy(mix.copy())


def separate(model, wave22: np.ndarray, wave_stereo: torch.Tensor):
    from demucs.apply import apply_model
    ref = wave_stereo.mean(0)
    x = (wave_stereo - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        src = apply_model(model, x[None].cuda(), device="cuda", split=True, overlap=0.25)[0]
    return src * (ref.std() + 1e-8) + ref.mean()


def to_mono22(t: torch.Tensor, sr: int) -> np.ndarray:
    import torchaudio
    return torchaudio.functional.resample(t.mean(0, keepdim=True), sr, features.SR)[0].cpu().numpy()


def main():
    from demucs.pretrained import get_model
    from chc import audio
    splits = sys.argv[1:] or ["train", "val", "test"]
    OUT.mkdir(parents=True, exist_ok=True)
    model = get_model("htdemucs_ft").cuda().eval()
    index = data.load_index()
    order = [(i, s) for i, s in enumerate(index) if data.split_of(s) in splits and "mix" in s["stems"]]
    # interleave so any prefix is a fair sample of the library
    rng = np.random.default_rng(0)
    rng.shuffle(order)
    for n, (i, s) in enumerate(order):
        need = [k for k in ("guitar", "bass", "drums") if k in s["notes"] and not (OUT / f"{i:04d}_{k}.npy").exists()]
        if not need:
            continue
        try:
            st = full_mix(s["stems"]["mix"])
            if st.shape[1] < 44100 * 20:
                continue
            src = separate(model, None, st)
            for name, srcs in zip(model.sources, src):
                kind = STEM_KIND.get(name)
                if kind in need:
                    w = to_mono22(srcs.cpu(), 44100)
                    np.save(OUT / f"{i:04d}_{kind}.npy", features.compute(w, kind))
            print(n, len(order), s["name"], flush=True)
        except Exception as e:  # keep going on bad files
            print("FAILED", s["name"], e, flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
