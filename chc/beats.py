"""Beat tracking: network beat/downbeat activations -> beat times (dynamic programming)."""
import numpy as np

from . import features


def local_period(act: np.ndarray, fps: float, lo=0.30, hi=0.90, win=8.0, step=2.0) -> np.ndarray:
    """Beat period (seconds) per frame from sliding-window autocorrelation of the activation."""
    n = len(act)
    w, h = int(win * fps), int(step * fps)
    lags = np.arange(int(lo * fps), int(hi * fps) + 1)
    prior = np.exp(-0.5 * (np.log2(lags / fps / 0.5) / 0.8) ** 2)  # favour ~120 BPM
    centers, periods = [], []
    for a in range(0, max(1, n - w + h), h):
        seg = act[a:a + w] - act[a:a + w].mean()
        if len(seg) < 2 * lags[-1]:
            break
        ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        score = np.array([ac[l] + 0.5 * ac[min(2 * l, len(ac) - 1)] for l in lags]) * prior
        periods.append(lags[int(np.argmax(score))] / fps)
        centers.append((a + len(seg) / 2) / fps)
    if not periods:
        return np.full(n, 0.5)
    periods = np.array(periods)
    k = 3  # median smoothing over neighbouring windows
    sm = np.array([np.median(periods[max(0, i - k):i + k + 1]) for i in range(len(periods))])
    return np.interp(np.arange(n) / fps, centers, sm)


def track(beat: np.ndarray, down: np.ndarray, fps: float = features.FPS, tightness=6.0,
          beats_per_bar=4):
    """Return (beat_times, downbeat_times) in seconds from per-frame probabilities."""
    act = np.convolve(beat, np.hanning(5) / np.hanning(5).sum(), mode="same")
    n = len(act)
    period = local_period(act, fps) * fps  # in frames
    score = np.full(n, -1e9)
    prev = np.full(n, -1, dtype=int)
    for i in range(n):
        p = period[i]
        j0, j1 = int(i - 1.5 * p), int(i - 0.6 * p)
        best, arg = 0.0, -1
        if j1 >= 0:
            js = np.arange(max(0, j0), j1 + 1)
            if len(js):
                pen = tightness * np.log((i - js) / p) ** 2
                cand = score[js] - pen
                k = int(np.argmax(cand))
                best, arg = cand[k], int(js[k])
        if arg >= 0 and best > 0:
            score[i], prev[i] = act[i] + best, arg
        else:
            score[i] = act[i]
    tail = int(max(1, 1.5 * period[-1]))
    i = int(np.argmax(score[-tail:]) + n - tail)
    path = []
    while i >= 0:
        path.append(i)
        i = prev[i]
    path = np.array(path[::-1])
    path = path[act[path] > 0.05] if len(path) > 8 else path
    beats = path / fps
    if len(path) == 0:
        return np.array([]), np.array([])
    dsum = [down[path[k::beats_per_bar]].sum() for k in range(beats_per_bar)]
    phase = int(np.argmax(dsum))
    return beats, beats[phase::beats_per_bar]
