"""Audio features for the note models: log-frequency spectrograms on a 256-sample hop.

Used both to build the training set and at inference time, so they must stay identical.
"""
import numpy as np
import torch

SR = 22050
HOP = 256
FPS = SR / HOP  # 86.13 frames per second

# kind -> (n_fft, n_bins, f_min, f_max, log-spaced?)
KINDS = {
    "guitar": (4096, 160, 60.0, 9000.0),
    "bass": (4096, 128, 28.0, 2500.0),
    "drums": (2048, 96, 25.0, 11000.0),
    "mix": (2048, 96, 25.0, 11000.0),  # full song, for beat tracking
    # sharper-in-time variants (shorter analysis window) for A/B tests
    "guitar2": (2048, 160, 60.0, 9000.0),
    "bass2": (2048, 128, 28.0, 2500.0),
}
_banks: dict = {}


def _bank(kind: str, device) -> torch.Tensor:
    key = (kind, str(device))
    if key not in _banks:
        n_fft, n_bins, lo, hi = KINDS[kind]
        freqs = np.fft.rfftfreq(n_fft, 1 / SR)
        centers = np.geomspace(lo, hi, n_bins + 2)
        fb = np.zeros((n_bins, len(freqs)), np.float32)
        for i in range(n_bins):
            l, c, r = centers[i], centers[i + 1], centers[i + 2]
            min_w = 2 * SR / n_fft  # never narrower than ~2 FFT bins
            l, r = min(l, c - min_w), max(r, c + min_w)
            up = (freqs - l) / (c - l)
            down = (r - freqs) / (r - c)
            fb[i] = np.clip(np.minimum(up, down), 0, None)
        fb /= np.maximum(fb.sum(axis=1, keepdims=True), 1e-9) / 2.0
        _banks[key] = torch.from_numpy(fb).to(device)
    return _banks[key]


def compute(wave: np.ndarray, kind: str, device="cuda") -> np.ndarray:
    """wave: mono float32 at SR. Returns (frames, bins) float16 log-magnitudes."""
    n_fft = KINDS[kind][0]
    x = torch.from_numpy(wave).to(device)
    win = torch.hann_window(n_fft, device=device)
    spec = torch.stft(x, n_fft, HOP, window=win, center=True, return_complex=True).abs()
    mel = _bank(kind, device) @ spec
    out = torch.log1p(30.0 * mel).T.contiguous()
    return out.half().cpu().numpy()


def normalize(f: np.ndarray) -> np.ndarray:
    """Per-song standardisation so quiet and loud mixes look alike."""
    f = f.astype(np.float32)
    return (f - f.mean()) / (f.std() + 1e-6)
