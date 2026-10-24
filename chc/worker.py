"""Stem separation worker, run as a subprocess so the GUI stays responsive and cancellable.

    python -m chc.worker <input.wav> <out_dir> [model]

Writes drums.wav, bass.wav, other.wav, vocals.wav to out_dir and prints
"PROGRESS <0..1>" lines on stdout.
"""
import re
import sys
from pathlib import Path

import numpy as np


class _TqdmSpy:
    """Receives tqdm's stderr output and turns '45%|' into PROGRESS lines."""

    def write(self, s):
        m = re.findall(r"(\d+)%\|", s)
        if m:
            print(f"PROGRESS {int(m[-1]) / 100:.2f}", flush=True)

    def flush(self):
        pass


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    model_name = sys.argv[3] if len(sys.argv) > 3 else "htdemucs_ft"
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from chc.audio import read_wav, write_wav
    import wave

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE {device}", flush=True)
    model = get_model(model_name)
    model.eval()
    with wave.open(str(src), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    wav = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, w.getnchannels()).T / 32768.0
    wav_t = torch.from_numpy(wav.copy())
    ref = wav_t.mean(0)
    wav_t = (wav_t - ref.mean()) / (ref.std() + 1e-8)
    sys.stderr = _TqdmSpy()
    with torch.no_grad():
        sources = apply_model(model, wav_t[None], device=device, split=True,
                              overlap=0.25, progress=True)[0]
    sources = sources * (ref.std() + 1e-8) + ref.mean()
    out.mkdir(parents=True, exist_ok=True)
    for name, s in zip(model.sources, sources):
        write_wav(out / f"{name}.wav", s.cpu().numpy(), model.samplerate)
    print("PROGRESS 1.00", flush=True)


if __name__ == "__main__":
    main()
