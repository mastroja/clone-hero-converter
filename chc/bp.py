"""Basic Pitch (Spotify, Apache-2.0) polyphonic note transcription via onnxruntime.

Only the bundled ONNX model is used; the rest of the basic-pitch package is not imported.
"""
import numpy as np

SR = 22050
HOP = 256
WIN = SR * 2 - HOP          # 43844 samples per model window
OVERLAP_FRAMES = 30
OVERLAP = OVERLAP_FRAMES * HOP
FPS = SR / HOP              # ~86.13
MIDI_LO = 21                # lowest model pitch (A0)

_session = None


def _model():
    global _session
    if _session is None:
        import importlib.util
        import os
        import onnxruntime as ort
        pkg = importlib.util.find_spec("basic_pitch").submodule_search_locations[0]
        path = os.path.join(pkg, "saved_models", "icassp_2022", "nmp.onnx")
        _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _session


def posteriors(x: np.ndarray, sr: int):
    """Return (note, onset) probability arrays shaped (frames, 88)."""
    if sr != SR:
        import torch, torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, SR)[0].numpy()
    x = np.concatenate([np.zeros(OVERLAP // 2, np.float32), x.astype(np.float32)])
    hop = WIN - OVERLAP
    n_win = max(1, int(np.ceil((len(x) - OVERLAP) / hop)))
    x = np.concatenate([x, np.zeros(n_win * hop + OVERLAP - len(x) + WIN, np.float32)])
    wins = np.stack([x[i * hop:i * hop + WIN] for i in range(n_win)])[..., None]
    sess = _model()
    out_names = [o.name for o in sess.get_outputs()]
    notes, onsets = [], []
    for k in range(0, n_win, 16):
        out = sess.run(None, {sess.get_inputs()[0].name: wins[k:k + 16]})
        d = dict(zip(out_names, out))
        notes.append(d["StatefulPartitionedCall:1"])
        onsets.append(d["StatefulPartitionedCall:2"])

    def unwrap(a):
        a = np.concatenate(a)
        a = a[:, OVERLAP_FRAMES // 2: -OVERLAP_FRAMES // 2, :]
        return a.reshape(-1, a.shape[2])
    return unwrap(notes), unwrap(onsets)


def attacks(note: np.ndarray, onset: np.ndarray, sensitivity=1.0, min_gap=0.07):
    """Collapse polyphonic output into [(time_s, pitch_height, strength)].

    Attack times are peaks of the summed onset posterior (robust to bends and
    vibrato). pitch_height is the activation-weighted mean MIDI pitch of the
    notes sounding just after the attack, which tracks melodic/chord height.
    """
    sal = np.clip(onset - 0.2, 0, None).sum(axis=1)
    sal = np.convolve(sal, np.hanning(5) / np.hanning(5).sum(), mode="same")
    w = int(0.6 * FPS)
    avg = np.convolve(sal, np.ones(2 * w + 1) / (2 * w + 1), mode="same")
    thr = np.maximum(avg * 1.3, sensitivity * 0.35 * np.percentile(sal, 99) + 1e-6)
    gap = int(min_gap * FPS)
    cand = np.where((sal[1:-1] >= sal[:-2]) & (sal[1:-1] > sal[2:]) & (sal[1:-1] > thr[1:-1]))[0] + 1
    keep = []
    for i in cand:
        if keep and i - keep[-1] < gap:
            if sal[i] > sal[keep[-1]]:
                keep[-1] = i
        else:
            keep.append(i)
    out = []
    for i in keep:
        h = height_at(note, i / FPS)
        if h is not None:
            out.append((i / FPS, h, float(sal[i])))
    return out


def height_at(note: np.ndarray, t: float):
    """Activation-weighted mean MIDI pitch of the notes sounding just after time t."""
    i = int(t * FPS)
    if i >= len(note):
        return None
    pitches = np.arange(note.shape[1]) + MIDI_LO
    seg = note[i + 1:i + 1 + int(0.25 * FPS)].mean(axis=0) if i + 2 < len(note) else note[i]
    act = np.where(seg > 0.25, seg, 0.0)
    act[(pitches < 40) | (pitches > 88)] = 0.0  # guitar lives roughly E2-E6
    if act.sum() <= 0:
        return None
    return float((act * pitches).sum() / act.sum())
