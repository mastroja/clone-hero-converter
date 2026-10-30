"""python -m train.feedback : how charts play, by the model that made them, from the ratings people
give songs in the app after playing them (chc.feedback, data/feedback.jsonl). Each song's latest
rating per part counts; model fingerprints are matched to the files in models/ and models/v3/."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from chc import feedback

MODELS = Path("models")


def model_names() -> dict:
    """fingerprint -> model file, for every saved model."""
    out = {}
    for p in sorted(MODELS.rglob("*.pt")):
        if ".resume" not in p.name:
            out.setdefault(hashlib.sha1(p.read_bytes()).hexdigest()[:8], str(p))
    return out


def main():
    if not feedback.LOG.exists():
        print("No ratings yet: rate songs in the app (song panel -> How does it play?)")
        return
    rows = [json.loads(line) for line in feedback.LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest, notes = {}, {}
    for r in rows:  # oldest first, so later ratings replace earlier ones
        if r["inst"]:
            tag = dict(m.split(":", 1) for m in r["models"].split()).get(r["inst"], "") if r["models"] else ""
            latest[(r["song"], r["inst"], tag)] = r["rating"]
        if r["note"] is not None:
            notes[(r["song"], r["models"])] = r["note"]
    names = model_names()
    by_model = defaultdict(lambda: defaultdict(list))
    for (song, inst, tag), rating in latest.items():
        if rating:
            by_model[(inst, tag)][rating].append(song)
    for (inst, tag), got in sorted(by_model.items()):
        n = sum(len(v) for v in got.values())
        who = names.get(tag, "unknown model file") if tag else "charted before model stamps"
        print(f"{inst:6s} {tag or '-':8s} {who}: {n} songs | "
              + " | ".join(f"{k} {len(got[k])} ({len(got[k]) / n:.0%})" for k in feedback.RATINGS))
        for song in got["bad"]:
            print(f"    bad: {song}")
    print("\nnotes:")
    for (song, _), note in sorted(notes.items()):
        if note:
            print(f"  {song}: {note}")


if __name__ == "__main__":
    main()
