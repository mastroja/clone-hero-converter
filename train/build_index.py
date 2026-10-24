"""python -m train.build_index <library_root> [out.json] [--v3]

--v3 also keeps five-lane drum charts (converted to four-lane) and writes data/v3/index.json
by default."""
import json
import sys
from collections import Counter
from pathlib import Path

from .charts import index_library

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    v3 = "--v3" in sys.argv
    root = Path(args[0])
    out = Path(args[1] if len(args) > 1 else "data/v3/index.json" if v3 else "data/index.json")
    songs = index_library(root, convert_five=v3)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(songs), encoding="utf-8")
    c = Counter(k for s in songs for k in s["notes"])
    n = {k: sum(len(s["notes"][k]) for s in songs if k in s["notes"]) for k in c}
    print(f"{len(songs)} songs indexed -> {out}")
    print("songs per instrument:", dict(c))
    print("notes per instrument:", n)
