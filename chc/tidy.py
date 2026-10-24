"""Clean up song folder names, e.g.
"Jimi Hendrix - Voodoo Child (Slight Return) (Live) (Harmonix)" -> "Jimi Hendrix - Voodoo Child".

Scanning never touches disk; apply() renames folders (and fixes the name in song.ini)
and writes an undo log next to the songs folder root.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

SONG_MARKERS = ("song.ini", "notes.chart", "notes.mid")
PARENS = re.compile(r"\s*\([^()]*\)")
BRACKETS = re.compile(r"\s*\[[^\[\]]*\]")
UNDO_LOG = ".chc_rename_log.json"


def clean_name(name: str, parens: bool = True, brackets: bool = True) -> str:
    """Remove (...) and/or [...] groups, then tidy spaces and dangling separators."""
    out = name
    for _ in range(3):  # nested groups
        before = out
        if parens:
            out = PARENS.sub("", out)
        if brackets:
            out = BRACKETS.sub("", out)
        if out == before:
            break
    out = re.sub(r"\s{2,}", " ", out).strip(" -_.")
    out = re.sub(r"\s+-\s*$", "", out).strip()
    return out if out else name  # never produce an empty name


@dataclass
class Proposal:
    path: Path
    new_name: str
    note: str = ""      # why it cannot be applied, if so
    selected: bool = True

    @property
    def ok(self) -> bool:
        return not self.note


def is_song_folder(p: Path) -> bool:
    return p.is_dir() and any((p / m).exists() for m in SONG_MARKERS)


def scan(root: Path, parens: bool = True, brackets: bool = True) -> list[Proposal]:
    """Find song folders under root whose names would change."""
    found: list[Proposal] = []
    taken: set[Path] = set()

    def walk(d: Path):
        try:
            kids = sorted(c for c in d.iterdir() if c.is_dir())
        except OSError:
            return
        for c in kids:
            if is_song_folder(c):
                new = clean_name(c.name, parens, brackets)
                if new != c.name:
                    target = c.with_name(new)
                    note = ""
                    if target.exists() and target != c:
                        note = "a folder with that name already exists"
                    elif target in taken:
                        note = "another song would get the same name"
                    else:
                        taken.add(target)
                    found.append(Proposal(c, new, note, selected=not note))
            else:
                walk(c)
    walk(root)
    return found


def _fix_ini(folder: Path, parens: bool, brackets: bool) -> None:
    ini = folder / "song.ini"
    if not ini.exists():
        return
    try:
        text = ini.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        m = re.match(r"(\s*name\s*=\s*)(.*?)(\r?\n?)$", line, re.I)
        if m:
            lines[i] = m.group(1) + clean_name(m.group(2), parens, brackets) + m.group(3)
    try:
        ini.write_text("".join(lines), encoding="utf-8")
    except OSError:
        pass


def apply(proposals: list[Proposal], root: Path, parens: bool = True, brackets: bool = True,
          fix_ini: bool = True) -> tuple[int, list[str]]:
    """Rename the selected proposals. Returns (renamed_count, error messages)."""
    done, errors, log = 0, [], []
    for p in proposals:
        if not (p.selected and p.ok):
            continue
        target = p.path.with_name(p.new_name)
        try:
            p.path.rename(target)
            log.append({"from": str(p.path), "to": str(target)})
            if fix_ini:
                _fix_ini(target, parens, brackets)
            done += 1
        except OSError as e:
            errors.append(f"{p.path.name}: {e}")
    if log:
        try:
            (root / UNDO_LOG).write_text(json.dumps(log, indent=2), encoding="utf-8")
        except OSError:
            pass
    return done, errors


def can_undo(root: Path) -> bool:
    return (root / UNDO_LOG).exists()


def undo(root: Path) -> tuple[int, list[str]]:
    """Reverse the last apply(). song.ini names are not restored."""
    log_path = root / UNDO_LOG
    done, errors = 0, []
    try:
        entries = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0, ["no undo information found"]
    for e in reversed(entries):
        try:
            Path(e["to"]).rename(e["from"])
            done += 1
        except OSError as ex:
            errors.append(f"{Path(e['to']).name}: {ex}")
    if not errors:
        log_path.unlink(missing_ok=True)
    return done, errors
