"""Turn separated stems into rough note lists (numpy only).

Everything here is a first-draft heuristic: onsets come from spectral flux,
guitar/bass pitch from a harmonic product spectrum, drums from frequency bands.
"""
import numpy as np

HOP = 512
N_FFT = 2048


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    n = 1 + max(0, (len(x) - size) // hop)
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[np.minimum(idx, len(x) - 1)]


def stft_mag(x: np.ndarray, n_fft=N_FFT, hop=HOP) -> np.ndarray:
    win = np.hanning(n_fft).astype(np.float32)
    mags = []
    for start in range(0, 1 + max(0, (len(x) - n_fft) // hop), 2048):  # chunked: low memory
        n = min(2048, 1 + (len(x) - n_fft) // hop - start)
        seg = x[start * hop: start * hop + (n - 1) * hop + n_fft]
        mags.append(np.abs(np.fft.rfft(_frames(seg, n_fft, hop) * win, axis=1)))
    return np.concatenate(mags).astype(np.float32)


def band_flux(mag: np.ndarray, sr: int, lo: float, hi: float, n_fft=N_FFT) -> np.ndarray:
    """Positive spectral flux of the log-magnitude within [lo, hi] Hz."""
    k0, k1 = int(lo * n_fft / sr), max(int(hi * n_fft / sr), int(lo * n_fft / sr) + 1)
    s = np.log1p(100.0 * mag[:, k0:k1])
    d = np.maximum(0, np.diff(s, axis=0, prepend=s[:1])).sum(axis=1)
    return d / (k1 - k0) ** 0.5


def pick_peaks(env: np.ndarray, sr: int, min_gap=0.08, sensitivity=1.0, hop=HOP,
               rel=0.0) -> np.ndarray:
    """Indices of onsets: local maxima clearly above a moving average."""
    if env.max() <= 0:
        return np.array([], dtype=int)
    w = int(0.5 * sr / hop)
    k = np.ones(2 * w + 1) / (2 * w + 1)
    avg = np.convolve(env, k, mode="same")
    thr = np.maximum(avg + sensitivity * 0.5 * env.std(), rel * np.percentile(env, 99))
    gap = max(1, int(min_gap * sr / hop))
    cand = np.where((env[1:-1] >= env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > thr[1:-1]))[0] + 1
    keep, last = [], -gap
    for i in cand:  # strongest-first within a gap window
        if i - last >= gap:
            keep.append(i)
            last = i
        elif env[i] > env[keep[-1]]:
            keep[-1] = i
            last = i
    return np.array(keep, dtype=int)


def estimate_tempo(env: np.ndarray, sr: int, hop=HOP) -> tuple[float, float]:
    """Return (bpm, offset_seconds) of the first beat for a steady-tempo song."""
    fps = sr / hop
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    lags = np.arange(len(ac))
    best, best_score = 120.0, -1.0
    for bpm in np.arange(70, 180, 0.5):
        lag = fps * 60.0 / bpm
        i = int(round(lag))
        if i + 1 >= len(ac):
            continue
        score = ac[i] + 0.5 * ac[min(2 * i, len(ac) - 1)]
        score *= np.exp(-0.5 * (np.log2(bpm / 120.0) / 0.9) ** 2)  # prefer 80-160
        if score > best_score:
            best, best_score = bpm, score
    # refine with a comb filter over a fine bpm/offset grid
    best_bpm, best_off, top = best, 0.0, -1.0
    t_end = len(env) / fps
    for bpm in np.arange(best - 3, best + 3.01, 0.1):
        spb = 60.0 / bpm
        for off in np.arange(0, spb, 0.01):
            beats = np.arange(off, t_end, spb)
            half = beats[:-1] + spb / 2
            ib = np.minimum((beats * fps).astype(int), len(env) - 1)
            ih = np.minimum((half * fps).astype(int), len(env) - 1)
            s = env[ib].sum() + 0.4 * env[ih].sum()
            if s > top:
                best_bpm, best_off, top = bpm, off, s
    return round(float(best_bpm), 1), float(best_off)


def _hps_pitch(mag_row: np.ndarray, sr: int, n_fft: int, fmin: float, fmax: float) -> float | None:
    k0, k1 = int(fmin * n_fft / sr), int(fmax * n_fft / sr)
    hps = mag_row[:k1].copy()
    for h in (2, 3):
        d = mag_row[::h][:k1]
        hps[:len(d)] *= d
        hps[len(d):] = 0
    seg = hps[k0:k1]
    if seg.size == 0 or seg.max() <= 0:
        return None
    k = k0 + int(np.argmax(seg))
    return k * sr / n_fft


def pitched_notes(x: np.ndarray, sr: int, fmin: float, fmax: float,
                  sensitivity=1.0, min_gap=0.09) -> list[tuple[float, float]]:
    """[(time_s, midi_pitch)] for a guitar/bass-like stem."""
    mag = stft_mag(x, N_FFT, HOP)
    env = band_flux(mag, sr, fmin, min(fmax * 6, sr / 2 - 1))
    peaks = pick_peaks(env, sr, min_gap=min_gap, sensitivity=sensitivity)
    big = 8192 if fmax <= 400 else 4096
    win = np.hanning(big).astype(np.float32)
    out = []
    for p in peaks:
        t = p * HOP / sr
        s = int((t + 0.03) * sr)
        seg = x[s:s + big]
        if len(seg) < big or np.sqrt(np.mean(seg ** 2)) < 1e-3:
            continue
        f = _hps_pitch(np.abs(np.fft.rfft(seg * win)), sr, big, fmin, fmax)
        if f:
            out.append((t, 69 + 12 * np.log2(f / 440.0)))
    return out


def quantile_lanes(pitches: list[float], lanes=5) -> list[int]:
    if not pitches:
        return []
    s = sorted(pitches)
    cuts = [s[int(len(s) * k / lanes)] for k in range(1, lanes)]
    return [sum(p >= c for c in cuts) for p in pitches]


def _band_energy(mag, sr, lo, hi, n_fft=N_FFT):
    k0, k1 = int(lo * n_fft / sr), max(int(hi * n_fft / sr), int(lo * n_fft / sr) + 1)
    return (mag[:, k0:k1] ** 2).sum(axis=1)


def drum_hits(x: np.ndarray, sr: int) -> tuple[list[tuple[float, str]], np.ndarray]:
    """[(time_s, 'K'|'S'|'H'|'C')] plus the full-band onset envelope (for tempo).

    Onsets come from the full-band flux; each onset is then labelled by which
    frequency bands jump in energy right after it.
    """
    mag = stft_mag(x, N_FFT, HOP)
    full = band_flux(mag, sr, 40, 12000)
    bands = {"K": _band_energy(mag, sr, 40, 110),
             "S": _band_energy(mag, sr, 180, 400) + _band_energy(mag, sr, 2000, 5000),
             "H": _band_energy(mag, sr, 6000, 14000)}
    floor = {c: np.percentile(e, 95) for c, e in bands.items()}
    min_gain = {"K": 6.0, "S": 5.0, "H": 4.0}
    min_level = {"K": 0.25, "S": 0.2, "H": 0.1}
    hits, hat_levels = [], []
    for i in pick_peaks(full, sr, min_gap=0.045, sensitivity=0.8):
        pre = slice(max(0, i - 4), max(1, i - 1))
        post = slice(i, i + 3)
        t = i * HOP / sr
        for c, e in bands.items():
            p, q = e[post].mean(), e[pre].mean() + 1e-9
            if p > min_level[c] * floor[c] and 10 * np.log10(p / q + 1e-9) > min_gain[c]:
                hits.append([t, c, float(p / (floor[c] + 1e-9))])
    # a hat that is far louder than a typical hat is a crash
    hat_vals = [h[2] for h in hits if h[1] == "H"]
    if hat_vals:
        cut = np.percentile(hat_vals, 95)
        last = -9.0
        for h in hits:
            if h[1] == "H" and h[2] >= cut and h[0] - last > 1.5:
                h[1] = "C"
                last = h[0]
    hits.sort()
    return [(t, c) for t, c, _ in hits], full


def guitar_attacks(x: np.ndarray, sr: int, max_per_sec: float = 3.2) -> list[tuple[float, float]]:
    """[(time_s, pitch_height)] for a polyphonic guitar-like stem.

    Attacks come from Basic Pitch's onset posteriors, plus any strong spectral-flux
    edges it missed (big strums). Pitch height is the weighted mean pitch of what is
    sounding after the attack, which gives a stable melodic/chord contour. The weakest
    attacks are dropped so the chart never gets denser than max_per_sec on average
    within any 4-second window (hand charts leave out ghost notes).
    """
    from . import bp
    note, onset = bp.posteriors(x, sr)
    att = bp.attacks(note, onset)
    sal = np.array([s for _, _, s in att]) if att else np.array([1.0])
    events = [(t, h, float((sal < s).mean())) for t, h, s in att]  # strength = rank 0..1
    mag = stft_mag(x, N_FFT, HOP)
    flux = band_flux(mag, sr, 80, 5000)
    have = np.array([t for t, _, _ in events]) if events else np.array([-9.0])
    for i in pick_peaks(flux, sr, min_gap=0.08, sensitivity=2.0):
        t = i * HOP / sr
        if np.min(np.abs(have - t)) > 0.06:
            h = bp.height_at(note, t)
            if h is not None:
                events.append((t, h, float((flux < flux[i]).mean())))
    events.sort()
    # density cap: within each 4 s window keep at most max_per_sec * 4 strongest
    cap = int(max_per_sec * 4)
    keep = set(range(len(events)))
    times = np.array([e[0] for e in events])
    for i in range(len(events)):
        j = np.where((times >= times[i] - 2) & (times <= times[i] + 2))[0]
        if len(j) > cap:
            order = sorted(j, key=lambda k: -events[k][2])
            if i not in order[:cap]:
                keep.discard(i)
    return [(events[i][0], events[i][1]) for i in sorted(keep)]


def local_lanes(heights: list[float], times: list[float], lanes=5, window=12.0) -> list[int]:
    """Lane per note from pitch height relative to the notes within +-window seconds.

    Notes at (nearly) the same height as the previous note keep its lane so repeated
    pitches do not wobble between frets.
    """
    n = len(heights)
    h = np.asarray(heights)
    t = np.asarray(times)
    out = []
    for i in range(n):
        m = (t >= t[i] - window) & (t <= t[i] + window)
        loc = np.sort(h[m])
        rank = np.searchsorted(loc, h[i], side="left") / max(1, len(loc) - 1)
        lane = min(lanes - 1, int(rank * lanes))
        if i and abs(h[i] - h[i - 1]) < 0.6 and t[i] - t[i - 1] < 1.0:
            lane = out[-1]
        out.append(lane)
    return out
