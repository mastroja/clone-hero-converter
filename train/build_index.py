"""python -m train.build_index <library_root> [out.json]"""
import json
import sys
from collections import Counter
from pathlib import Path

from .charts import index_library

if __name__ == "__main__":
    root = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "data/index.json")
    songs = index_library(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(songs), encoding="utf-8")
    c = Counter(k for s in songs for k in s["notes"])
    n = {k: sum(len(s["notes"][k]) for s in songs if k in s["notes"]) for k in c}
    print(f"{len(songs)} songs indexed -> {out}")
    print("songs per instrument:", dict(c))
    print("notes per instrument:", n)
