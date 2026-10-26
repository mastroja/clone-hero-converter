"""Clean up song folder names, e.g.
"Jimi Hendrix - Voodoo Child (Slight Return) (Live) (Harmonix)" -> "Jimi Hendrix - Voodoo Child",
and (with from_ini) give every song the same "Artist - Song Title" form, taken from the artist
and name in its song.ini ("voodoo_child_final" -> "Jimi Hendrix - Voodoo Child").

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
BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')  # not allowed in Windows folder names


def read_ini(folder: Path) -> dict:
    out = {}
    try:
        text = (folder / "song.ini").read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$", line)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def _key(name: str) -> str:
    """Name for comparing: no capitals, '&' as 'and', letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", name.lower().replace("&", "and"))


def ini_name(folder: Path) -> str | None:
    """'Artist - Song Title' from the folder's song.ini, or None without both."""
    ini = read_ini(folder)
    artist, name = ini.get("artist", "").strip(), ini.get("name", "").strip()
    if not artist or not name:
        return None
    if name.lower().startswith(artist.lower() + " - "):
        return name  # the title already starts with the artist
    return f"{artist} - {name}"


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


def proposed_name(c: Path, parens: bool = True, brackets: bool = True, from_ini: bool = True) -> str:
    """The tidy folder name for one song folder (the same rules as scan)."""
    cleaned = clean_name(c.name, parens, brackets)
    base = (ini_name(c) if from_ini else None) or cleaned
    new = clean_name(base, parens, brackets)
    if _key(new) == _key(cleaned):
        new = cleaned  # same words: keep the folder's own spelling and capitals
    # Windows forbids some characters, and trailing dots or spaces
    return BAD_CHARS.sub("", new).rstrip(" .").strip() or c.name


def tidy_one(folder: Path, parens: bool = True, brackets: bool = True) -> Path:
    """Tidy a single song right away (after its details are edited): clean the title in
    song.ini and rename the folder 'Artist - Song Title'. Raises FileExistsError if that
    name is taken. Returns the folder's new path."""
    _fix_ini(folder, parens, brackets)
    target = folder.with_name(proposed_name(folder, parens, brackets))
    if target != folder:
        if target.exists() and target.resolve() != folder.resolve():
            raise FileExistsError(target.name)
        folder.rename(target)  # a case-only rename works on Windows too
    return target


def scan(root: Path, parens: bool = True, brackets: bool = True, from_ini: bool = True) -> list[Proposal]:
    """Find song folders under root whose names would change. from_ini names each song
    "Artist - Song Title" from its song.ini (when it has both); otherwise the folder name is
    only cleaned."""
    found: list[Proposal] = []
    taken: set[Path] = set()

    def walk(d: Path):
        try:
            kids = sorted(c for c in d.iterdir() if c.is_dir())
        except OSError:
            return
        for c in kids:
            if is_song_folder(c):
                new = proposed_name(c, parens, brackets, from_ini)
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


def write_ini(folder: Path, fields: dict) -> None:
    """Set song.ini keys (replacing existing lines, adding the rest under [song]); other lines kept."""
    ini = folder / "song.ini"
    try:
        text = ini.read_text(encoding="utf-8-sig")
    except OSError:
        text = "[song]\n"
    for key, value in fields.items():
        value = " ".join(str(value).split())  # one line
        pat = re.compile(rf"(?mi)^(\s*{re.escape(key)}\s*=).*$")
        if pat.search(text):
            text = pat.sub(lambda m: f"{m.group(1)} {value}", text, count=1)
        elif re.search(r"(?mi)^\s*\[song\]\s*$", text):
            text = re.sub(r"(?mi)^(\s*\[song\]\s*)$", lambda m: f"{m.group(1)}\n{key} = {value}", text, count=1)
        else:
            text = text.rstrip("\n") + f"\n{key} = {value}\n"
    ini.write_text(text, encoding="utf-8")


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
