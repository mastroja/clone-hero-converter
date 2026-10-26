"""RockWizard's local server: the desktop window and the phone web app both talk to it.

Runs on this PC (its GPU does the work). Requests from this PC are always allowed; from other
devices (a phone over the home network or Tailscale) only with the pairing key, which the phone
gets once from the QR code / link in Settings and then keeps as a cookie.

    python -m chc.server            (http://127.0.0.1:8765; the desktop app starts it itself)
"""
import json
import os
import queue
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import library, pipeline, review, search, tidy

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
CONFIG = Path.home() / ".clone_hero_converter.json"  # shared with the earlier desktop app
PORT = 8765
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma"}
STEM_NAMES = {"bass", "drums", "other", "vocals", "guitar", "piano", "keys", "song", "crowd", "preview",
              "rhythm", "drums_1", "drums_2", "drums_3", "drums_4", "no_vocals", "instrumental"}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ---------- settings ----------
def load_config() -> dict:
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    changed = False
    if not cfg.get("songs_dir"):
        cfg["songs_dir"] = str(Path.home() / "Documents" / "Clone Hero" / "Songs")
        changed = True
    if not cfg.get("pair_key"):
        cfg["pair_key"] = secrets.token_urlsafe(16)
        changed = True
    songs = Path(cfg["songs_dir"])
    # guesses for a usual Clone Hero install: <Clone Hero>\songs\<staging>, \songs\Quickplay
    if not cfg.get("live_dir") and (songs.parent / "Quickplay").is_dir() and songs.name != "Quickplay":
        cfg["live_dir"] = str(songs.parent / "Quickplay")
        changed = True
    if not cfg.get("clonehero_exe"):
        for p in [songs, *songs.parents][:4]:
            if (p / "Clone Hero.exe").exists():
                cfg["clonehero_exe"] = str(p / "Clone Hero.exe")
                changed = True
                break
    if changed:
        save_config(cfg)
    return cfg


def save_config(cfg: dict) -> None:
    try:
        CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except OSError:
        pass


cfg = load_config()


def staging_dir() -> Path:
    return Path(cfg.get("songs_dir"))


def live_dir() -> Path | None:
    return Path(cfg["live_dir"]) if cfg.get("live_dir") else None


# ---------- jobs ----------
@dataclass
class Job:
    id: str
    kind: str                 # "convert" or "update"
    title: str
    steps: list
    status: str = "queued"    # queued, running, done, failed, cancelled
    step: int = -1
    progress: float = 0.0
    message: str = ""
    result: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    args: dict = field(default_factory=dict)


jobs: dict[str, Job] = {}
job_queue: "queue.Queue[str]" = queue.Queue()
stops: dict[str, threading.Event] = {}
CONVERT_STEPS = ["Downloading the song"] + pipeline.STEPS


def add_job(kind, title, steps, **args) -> Job:
    j = Job(uuid.uuid4().hex[:10], kind, title, steps, args=args)
    jobs[j.id] = j
    stops[j.id] = threading.Event()
    job_queue.put(j.id)
    return j


def public(j: Job) -> dict:
    d = asdict(j)
    d.pop("args")
    return d


def run_convert(j: Job, stop: threading.Event):
    a = j.args
    tmp = Path(tempfile.mkdtemp(prefix="rw_job_"))
    try:
        src, offset = a.get("path"), 0
        if a.get("url"):
            j.step = 0
            src = search.youtube_download(a["url"], tmp, lambda f: setattr(j, "progress", 0.12 * f), stop)
            offset = 1
        artist, title = a.get("artist", ""), a.get("title", "") or Path(src).stem
        details = search.lookup(artist, title)
        if details and search.similar(details.title, title) >= 0.8:
            artist, title = details.artist, details.title  # proper spelling, fixes swapped names
        j.title = f"{artist + ' - ' if artist else ''}{title}"
        cover = search.cover_art(details, tmp / "cover.jpg") if details else None
        start = 0.12 if a.get("url") else 0.0

        def on_step(i):
            j.step = i + offset

        def on_progress(f):
            j.progress = start + (1 - start) * f
        res = pipeline.make_song(Path(src), artist, title, staging_dir(), charter=cfg.get("charter", ""),
                                 on_step=on_step, on_progress=on_progress, stop=stop,
                                 details=details, cover=cover)
        shutil.rmtree(res.preview_dir, ignore_errors=True)
        rep = review.check(res.song_dir, live_dir())
        j.result = {"folder": str(res.song_dir), "counts": res.counts, "verdict": rep.verdict,
                    "replaced": [str(p) for p in res.replaced]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_update(j: Job, stop: threading.Event):
    folder = Path(j.args["folder"])
    saved = pipeline.update_song(folder, on_step=lambda i: setattr(j, "step", i),
                                 on_progress=lambda f: setattr(j, "progress", f), stop=stop)
    rep = review.check(folder, live_dir())
    j.result = {"folder": str(folder), "backup": str(saved), "verdict": rep.verdict}


def worker():
    while True:
        jid = job_queue.get()
        j = jobs.get(jid)
        if j is None or j.status == "cancelled":
            continue
        j.status = "running"
        try:
            (run_convert if j.kind == "convert" else run_update)(j, stops[jid])
            j.status, j.progress, j.step = "done", 1.0, len(j.steps)
        except pipeline.Cancelled:
            j.status = "cancelled"
        except Exception as e:
            traceback.print_exc()
            j.status = "cancelled" if stops[jid].is_set() else "failed"
            j.message = str(e).splitlines()[0] if str(e) else e.__class__.__name__


threading.Thread(target=worker, daemon=True).start()


# ---------- app + access ----------
app = FastAPI(title="RockWizard")


@app.middleware("http")
async def access(request: Request, call_next):
    host = request.client.host if request.client else ""
    local = host in ("127.0.0.1", "::1", "localhost")
    key = request.query_params.get("key") or request.cookies.get("rw_key")
    if not local and key != cfg["pair_key"]:
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "This device isn't paired. Scan the QR code in RockWizard's Settings."},
                                status_code=401)
        return HTMLResponse("<h2 style='font-family:system-ui;padding:24px'>RockWizard: this device isn't paired."
                            "<br>Open Settings &rarr; Phone in RockWizard on your PC and scan the QR code.</h2>",
                            status_code=401)
    response = await call_next(request)
    if not local and request.query_params.get("key") == cfg["pair_key"]:
        response.set_cookie("rw_key", cfg["pair_key"], max_age=10 * 365 * 86400, samesite="lax")
    return response


def body_folder(path: str, base: Path | None = None) -> Path:
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, f"{p} not found")
    return p


# ---------- jobs API ----------
@app.get("/api/jobs")
def list_jobs():
    return [public(j) for j in sorted(jobs.values(), key=lambda j: -j.created)][:100]


@app.post("/api/jobs/upload")
async def job_upload(file: UploadFile = File(...), artist: str = Form(""), title: str = Form("")):
    up = Path(tempfile.mkdtemp(prefix="rw_up_")) / Path(file.filename).name
    with open(up, "wb") as f:
        shutil.copyfileobj(file.file, f)
    a, t = search.names_from_file(up.stem)
    j = add_job("convert", f"{artist or a} - {title or t}".strip(" -"), CONVERT_STEPS[1:],
                path=str(up), artist=artist or a, title=title or t)
    return public(j)


@app.post("/api/jobs/convert")
async def job_convert(req: Request):
    """{path} for a file on this PC, or {url} for YouTube; optional artist/title."""
    b = await req.json()
    if b.get("url"):
        steps = CONVERT_STEPS
        if not b.get("title"):  # a pasted link: names from the video's title and channel
            try:
                b["artist"], b["title"] = search.names_from_video(*search.video_title(b["url"]))
            except Exception:
                pass
    else:
        if not Path(b.get("path", "")).exists():
            raise HTTPException(404, "file not found")
        steps = CONVERT_STEPS[1:]
        if not b.get("title"):
            b["artist"], b["title"] = search.names_from_file(Path(b["path"]).stem)
    j = add_job("convert", f"{b.get('artist', '')} - {b.get('title', '')}".strip(" -") or "YouTube song",
                steps, path=b.get("path"), url=b.get("url"), artist=b.get("artist", ""), title=b.get("title", ""))
    return public(j)


@app.post("/api/jobs/update")
async def job_update(req: Request):
    b = await req.json()
    out = []
    for f in b.get("folders", []):
        if library.is_ours(Path(f)):
            out.append(public(add_job("update", Path(f).name, pipeline.UPDATE_STEPS, folder=f)))
    return out


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404)
    stops[jid].set()
    if j.status == "queued":
        j.status = "cancelled"
    return public(j)


@app.post("/api/jobs/clear")
def jobs_clear():
    for jid in [k for k, j in jobs.items() if j.status in ("done", "failed", "cancelled")]:
        jobs.pop(jid, None)
    return {"ok": True}


# ---------- search ----------
@app.get("/api/search")
def do_search(q: str):
    artist, title = search.parse_query(q)
    out = {"charts": [], "details": None, "videos": [], "errors": []}
    try:
        out["charts"] = [asdict(c) | {"download_url": c.download_url} for c in search.community_charts(artist, title)]
    except Exception as e:
        out["errors"].append(f"chart search ({e.__class__.__name__})")
    d = search.lookup(artist, title)
    if d:
        out["details"] = asdict(d)
        artist = artist or d.artist
    try:
        vids = search.youtube_search(artist, title, expect_seconds=d.seconds if d else 0)
        for v in vids:
            va, vt = search.names_from_video(v.title, v.channel)
            if d and search.similar(d.title, vt) >= 0.8:
                va, vt = d.artist, d.title
            out["videos"].append(asdict(v) | {"artist": va or artist, "song": vt or title})
    except Exception as e:
        out["errors"].append(f"YouTube search ({e.__class__.__name__}; try Settings, Update downloader)")
    return out


@app.post("/api/charts/download")
async def chart_download(req: Request):
    b = await req.json()
    c = search.Chart(b["name"], b["artist"], b.get("charter", "unknown"), b["md5"])
    dest = live_dir() if b.get("to") == "live" and live_dir() else staging_dir()
    return {"saved": str(search.download_chart(c, dest))}


# ---------- staging ----------
_reviews: dict[str, dict] = {}


def review_dict(folder: Path) -> dict:
    rep = review.check(folder, live_dir())
    d = {"verdict": rep.verdict, "findings": [asdict(f) for f in rep.findings],
         "checked": time.time(), "mtime": (folder / "notes.mid").stat().st_mtime if (folder / "notes.mid").exists() else 0}
    _reviews[str(folder)] = d
    return d


def song_summary(folder: Path, with_review=False) -> dict:
    ini = tidy.read_ini(folder)
    r = _reviews.get(str(folder))
    mtime = (folder / "notes.mid").stat().st_mtime if (folder / "notes.mid").exists() else 0
    if with_review and (r is None or r.get("mtime") != mtime):
        r = review_dict(folder)
    return {"folder": str(folder), "name": folder.name, "artist": ini.get("artist", ""), "title": ini.get("name", ""),
            "album": ini.get("album", ""), "year": ini.get("year", ""), "status": review.status(folder),
            "tiers": {k: ini.get(f"diff_{k}", "") for k in ("guitar", "bass", "drums")},
            "has_cover": (folder / "album.jpg").exists() or (folder / "album.png").exists(),
            "stems": [p.stem for p in folder.iterdir() if p.suffix.lower() in (".ogg", ".opus", ".mp3")],
            "modified": mtime, "verdict": r["verdict"] if r else None}


@app.get("/api/staging")
def staging_list(review_all: bool = False):
    root = staging_dir()
    if not root.exists():
        return {"folder": str(root), "songs": []}
    songs = [song_summary(p, review_all) for p in library.song_folders(root) if library.is_ours(p)]
    return {"folder": str(root), "songs": sorted(songs, key=lambda s: -s["modified"])}


@app.get("/api/staging/review")
def staging_review(folder: str):
    f = body_folder(folder)
    return song_summary(f) | review_dict(f)


@app.post("/api/staging/status")
async def staging_status(req: Request):
    b = await req.json()
    review.set_status(body_folder(b["folder"]), b["status"])
    return {"ok": True}


@app.post("/api/staging/publish")
async def staging_publish(req: Request):
    b = await req.json()
    if not live_dir():
        raise HTTPException(400, "Set the live folder (Quickplay) in Settings first")
    ok, msg, dest = review.publish(body_folder(b["folder"]), live_dir(), bool(b.get("force")))
    return {"ok": ok, "message": msg, "folder": str(dest) if dest else None}


@app.post("/api/open")
async def open_on_pc(req: Request):
    """Desktop helpers: open a folder, play in Clone Hero, open in Moonscraper (on this PC)."""
    b = await req.json()
    what, folder = b.get("what"), body_folder(b["folder"])
    if what == "folder":
        os.startfile(folder)
    elif what == "clonehero":
        exe = cfg.get("clonehero_exe", "")
        if not exe or not Path(exe).exists():
            raise HTTPException(400, "Set Clone Hero's location in Settings")
        subprocess.Popen([exe], cwd=str(Path(exe).parent))
    elif what == "moonscraper":
        exe = cfg.get("moonscraper_exe", "")
        if not exe or not Path(exe).exists():
            raise HTTPException(400, "Set Moonscraper's location in Settings")
        subprocess.Popen([exe, str(folder / "notes.mid")], cwd=str(Path(exe).parent))
    return {"ok": True}


@app.get("/api/audio")
def stem_audio(folder: str, stem: str):
    f = body_folder(folder)
    for ext in (".ogg", ".opus", ".mp3"):
        p = f / f"{stem}{ext}"
        if p.exists():
            return FileResponse(p, media_type="audio/ogg" if ext != ".mp3" else "audio/mpeg")
    raise HTTPException(404)


@app.get("/api/cover")
def cover(folder: str):
    f = body_folder(folder)
    for n in ("album.jpg", "album.png"):
        if (f / n).exists():
            return FileResponse(f / n)
    raise HTTPException(404)


# ---------- library ----------
@app.get("/api/library/ours")
def library_ours(folder: str = ""):
    root = Path(folder) if folder else staging_dir()
    return {"folder": str(root), "songs": [song_summary(p) for p in library.song_folders(root) if library.is_ours(p)]}


@app.get("/api/library/batch")
def library_batch(folder: str):
    root = body_folder(folder)
    have = {}
    if staging_dir().exists():
        for f in library.song_folders(staging_dir()):
            if library.is_ours(f):
                have[library.song_key(*library.names_of(f))] = f.name
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT
                   and not tidy.is_song_folder(p.parent) and p.stem.lower() not in STEM_NAMES)
    out = []
    for p in files:
        a, t = search.names_from_file(p.stem)
        out.append({"path": str(p), "file": p.name, "artist": a, "title": t,
                    "existing": have.get(library.song_key(a, t)) or have.get(library.song_key(t, a))})
    return {"folder": str(root), "files": out}


@app.get("/api/tidy")
def tidy_scan(folder: str, parens: bool = True, brackets: bool = True, ini: bool = True):
    root = body_folder(folder)
    props = tidy.scan(root, parens, brackets, ini)
    return {"folder": str(root), "can_undo": tidy.can_undo(root),
            "proposals": [{"path": str(p.path), "name": p.path.name, "new_name": p.new_name, "note": p.note}
                          for p in props]}


@app.post("/api/tidy/apply")
async def tidy_apply(req: Request):
    b = await req.json()
    root = body_folder(b["folder"])
    chosen = set(b.get("paths", []))
    props = [p for p in tidy.scan(root, b.get("parens", True), b.get("brackets", True), b.get("ini", True))]
    for p in props:
        p.selected = str(p.path) in chosen
    n, errs = tidy.apply(props, root, b.get("parens", True), b.get("brackets", True))
    return {"renamed": n, "errors": errs}


@app.post("/api/tidy/undo")
async def tidy_undo(req: Request):
    b = await req.json()
    n, errs = tidy.undo(body_folder(b["folder"]))
    return {"restored": n, "errors": errs}


@app.get("/api/dupes")
def dupes(folder: str):
    root = body_folder(folder)
    groups = library.find_duplicates(root)
    return {"folder": str(root), "groups": [[{
        "path": str(c.folder), "name": c.folder.name, "artist": c.artist, "title": c.title, "charter": c.charter,
        "parts": c.parts, "size_mb": round(c.size_mb, 1), "ours": c.ours, "keep": c.keep,
        "identical": c.identical} for c in g] for g in groups]}


@app.post("/api/dupes/remove")
async def dupes_remove(req: Request):
    b = await req.json()
    n, errs = library.recycle([Path(p) for p in b.get("paths", []) if Path(p).exists()])
    return {"removed": n, "errors": errs}


# ---------- settings + status ----------
SETTING_KEYS = ("songs_dir", "live_dir", "tidy_dir", "charter", "clonehero_exe", "moonscraper_exe")


@app.get("/api/settings")
def get_settings(request: Request):
    local = request.client and request.client.host in ("127.0.0.1", "::1")
    return {k: cfg.get(k, "") for k in SETTING_KEYS} | {"local": bool(local)}


@app.post("/api/settings")
async def set_settings(req: Request):
    b = await req.json()
    for k in SETTING_KEYS:
        if k in b:
            cfg[k] = str(b[k]).strip()
    save_config(cfg)
    return {k: cfg.get(k, "") for k in SETTING_KEYS}


@app.post("/api/downloader/update")
def downloader_update():
    return {"message": search.update_downloader()}


def lan_addresses() -> list[str]:
    import socket
    addrs = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                addrs.add(ip)
    except OSError:
        pass
    # Tailscale addresses (100.64.0.0/10) first: they work away from home
    return sorted(addrs, key=lambda ip: (not ip.startswith("100."), ip))


@app.get("/api/pairing")
def pairing():
    import io
    import qrcode
    import qrcode.image.svg
    urls = [f"http://{ip}:{PORT}/?key={cfg['pair_key']}" for ip in lan_addresses()]
    svg = ""
    if urls:
        buf = io.BytesIO()
        qrcode.make(urls[0], image_factory=qrcode.image.svg.SvgPathImage, box_size=10).save(buf)
        svg = buf.getvalue().decode()
    return {"urls": urls, "qr": svg}


@app.post("/api/pairing/reset")
def pairing_reset():
    cfg["pair_key"] = secrets.token_urlsafe(16)
    save_config(cfg)
    return pairing()


def syncthing_status() -> dict | None:
    """How the live folder is syncing (Syncthing's local REST API), or None without Syncthing."""
    conf = Path(os.environ.get("LOCALAPPDATA", "")) / "Syncthing" / "config.xml"
    if not conf.exists() or not live_dir():
        return None
    try:
        import urllib.request
        root = ET.parse(conf).getroot()
        key = root.findtext("gui/apikey")
        addr = root.findtext("gui/address") or "127.0.0.1:8384"
        live = str(live_dir().resolve()).lower()
        folder = next((f for f in root.findall("folder") if str(Path(f.get("path")).resolve()).lower() == live), None)
        if folder is None:
            return {"state": "not shared", "label": ""}

        def get(path):
            r = urllib.request.Request(f"http://{addr}{path}", headers={"X-API-Key": key})
            return json.load(urllib.request.urlopen(r, timeout=3))
        st = get(f"/rest/db/status?folder={folder.get('id')}")
        devices = [d.get("id") for d in folder.findall("device")]
        names = {d.get("id"): d.get("name") for d in root.findall("device")}
        conns = get("/rest/system/connections").get("connections", {})
        remote = []
        for d in devices:
            if d not in conns:
                continue
            comp = get(f"/rest/db/completion?folder={folder.get('id')}&device={d}")
            remote.append({"name": names.get(d, "device"), "connected": conns[d].get("connected", False),
                           "completion": round(comp.get("completion", 0), 1)})
        return {"state": st.get("state", "?"), "label": folder.get("label", ""), "devices": remote}
    except Exception as e:
        return {"state": "unavailable", "error": e.__class__.__name__}


@app.post("/api/sync/rescan")
def sync_rescan():
    conf = Path(os.environ.get("LOCALAPPDATA", "")) / "Syncthing" / "config.xml"
    try:
        import urllib.request
        root = ET.parse(conf).getroot()
        key, addr = root.findtext("gui/apikey"), root.findtext("gui/address") or "127.0.0.1:8384"
        live = str(live_dir().resolve()).lower()
        folder = next(f for f in root.findall("folder") if str(Path(f.get("path")).resolve()).lower() == live)
        r = urllib.request.Request(f"http://{addr}/rest/db/scan?folder={folder.get('id')}", method="POST",
                                   headers={"X-API-Key": key})
        urllib.request.urlopen(r, timeout=5)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(400, f"Couldn't reach Syncthing ({e.__class__.__name__})")


@app.get("/api/status")
def status():
    run = ROOT / "data" / "v3" / "run.log"
    training = None
    if run.exists():
        lines = run.read_text(errors="replace").strip().splitlines()
        last = lines[-1] if lines else ""
        lock = ROOT / "data" / "v3" / "RUNNING"
        training = {"last": last, "running": lock.exists() and "done" not in last.split(" ")[-8:]}
    busy = any(j.status == "running" for j in jobs.values())
    queued = sum(1 for j in jobs.values() if j.status == "queued")
    return {"busy": busy, "queued": queued, "training": training, "sync": syncthing_status()}


# ---------- web app ----------
@app.get("/")
def index():
    # stamp the script and styles with their change time so browsers (and the phone's home-screen
    # app) always load the current version after an update
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for name in ("app.js", "app.css"):
        html = html.replace(f"/static/{name}", f"/static/{name}?v={int((WEB / name).stat().st_mtime)}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/manifest.webmanifest")
def manifest():
    return Response((WEB / "manifest.webmanifest").read_text(encoding="utf-8"),
                    media_type="application/manifest+json")


app.mount("/static", StaticFiles(directory=WEB), name="static")


def serve(host="0.0.0.0", port=PORT):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    serve()
