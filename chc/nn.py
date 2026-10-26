"""Convolutional + recurrent note-onset network shared by training and the app."""
import numpy as np
import torch
from torch import nn

from . import features

DOWN = 4  # the recurrent layer runs at 1/4 of the frame rate

# output channels per instrument; guitar/bass add a hammer-on flag (at the note) and a
# sustain-held channel (on while a sustain rings) after the five lanes
CHANNELS = {
    "guitar": ["G", "R", "Y", "B", "O", "hopo", "sus"],
    "bass": ["G", "R", "Y", "B", "O", "hopo", "sus"],
    "drums": ["K", "S", "Ycym", "Ytom", "Bcym", "Btom", "Gcym", "Gtom"],
}
CHANNELS["mix"] = ["beat", "down"]
for _k in ("guitar2", "guitar_hr"):
    CHANNELS[_k] = CHANNELS["guitar"]
for _k in ("bass2", "bass_hr"):
    CHANNELS[_k] = CHANNELS["bass"]
# optional extra outputs (train --rel): is this note's lowest fret above, equal to or below
# the previous note's? Charters place frets by the melody's shape, not absolute pitch.
REL = ["up", "same", "down"]
# optional extra outputs after those (train --size): how many frets the note has (1, 2, 3+);
# chord size was the biggest guitar error when chords were picked from the lane outputs alone
SIZE = ["1 fret", "2 frets", "3 frets"]


def instrument(kind: str) -> str:
    """Feature/model kind -> instrument ('guitar_hr' -> 'guitar', 'bass2' -> 'bass')."""
    return kind.split("_")[0].rstrip("2")


def n_lanes(kind: str) -> int:
    """Number of note-onset channels (the rest are per-note attributes)."""
    return 5 if instrument(kind) in ("guitar", "bass") else len(CHANNELS[kind])


class NoteNet(nn.Module):
    def __init__(self, kind: str, hidden: int = 128, n_out: int | None = None, context: bool = False):
        """context: also take the whole song's feature profile (features.song_profile) next to
        every window, so the model knows where the music sits in the song's range."""
        super().__init__()
        self.kind = kind
        self.n_out = n_out or len(CHANNELS[kind])
        self.context = context
        n_bins = features.n_bins(kind)
        ch = [1, 16, 32, 48]
        pools = [3, 3, 2]
        layers, f = [], n_bins
        for i in range(3):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]), nn.ReLU(),
                       nn.Conv2d(ch[i + 1], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]), nn.ReLU(),
                       nn.MaxPool2d((1, pools[i]))]
            f //= pools[i]
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Linear(ch[-1] * f, 256), nn.ReLU(), nn.Dropout(0.2))
        width = 256
        if context:
            self.ctx = nn.Sequential(nn.Linear(2 * n_bins, 64), nn.ReLU())
            width += 64
        self.rnn = nn.GRU(width, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.out = nn.Linear(2 * hidden + width, self.n_out)

    def forward(self, x, profile=None):  # x: (B, T, F) normalised features; profile: (B, 2F)
        h = self.conv(x.unsqueeze(1))                      # (B, C, T, F')
        h = h.permute(0, 2, 1, 3).flatten(2)               # (B, T, C*F')
        h = self.proj(h)                                   # full time resolution
        if self.context:
            c = self.ctx(profile)[:, None, :].expand(-1, h.shape[1], -1)
            h = torch.cat([h, c], dim=-1)
        T = h.shape[1]
        pad = (-T) % DOWN
        hp = nn.functional.pad(h, (0, 0, 0, pad))
        ctx = nn.functional.avg_pool1d(hp.transpose(1, 2), DOWN).transpose(1, 2)
        ctx, _ = self.rnn(ctx)                             # long-range context at 1/DOWN rate
        ctx = ctx.repeat_interleave(DOWN, dim=1)[:, :T]
        return self.out(torch.cat([ctx, h], dim=-1))       # logits (B, T, n_out)


@torch.no_grad()
def predict(model: NoteNet, feats: np.ndarray, device="cuda", chunk=4096, pad=256) -> np.ndarray:
    """Probabilities (T, n_out) for a whole song, processed in overlapping chunks."""
    model.eval()
    xn = features.normalize(feats)
    x = torch.from_numpy(xn).to(device)
    prof = torch.from_numpy(features.song_profile(xn))[None].to(device) if model.context else None
    T = x.shape[0]
    out = np.zeros((T, model.n_out), np.float32)
    start = 0
    while start < T:
        a, b = max(0, start - pad), min(T, start + chunk + pad)
        p = torch.sigmoid(model(x[a:b][None], prof))[0].cpu().numpy()
        out[start:min(T, start + chunk)] = p[start - a: start - a + min(chunk, T - start)]
        start += chunk
    return out


def pick(prob: np.ndarray, thresholds, min_gap_frames=3) -> list[tuple[int, int, float]]:
    """Local maxima above per-channel threshold -> [(frame, channel, prob)] sorted by frame."""
    hits = []
    for c in range(prob.shape[1]):
        p = prob[:, c]
        thr = thresholds[c] if hasattr(thresholds, "__len__") else thresholds
        idx = np.where((p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] >= thr))[0] + 1
        last = -10 ** 9
        for i in idx:
            if i - last >= min_gap_frames:
                hits.append((int(i), c, float(p[i])))
                last = i
            elif hits and p[i] > hits[-1][2] and hits[-1][1] == c:
                hits[-1] = (int(i), c, float(p[i]))
                last = i
    return sorted(hits)
