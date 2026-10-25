#!/usr/bin/env python
"""Clone Hero Converter: drop in a song, get a Clone Hero song folder."""
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import traceback
import webbrowser
from pathlib import Path

import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD

from chc import pipeline, search, tidy

try:
    import winsound
except ImportError:  # non-Windows: previews disabled
    winsound = None

CONFIG = Path.home() / ".clone_hero_converter.json"
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma"}
ACCENT = "#7c5cff"
ACCENT_HOVER = "#6a49ec"
GOOD = "#35c486"
MUTED = ("#6b7280", "#9aa0ad")
CARD = ("#eef0f6", "#1d2030")
CARD_HOVER = ("#e2e5f0", "#262a3f")


def load_config() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    try:
        CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except OSError:
        pass


def short(text: str, n: int = 52) -> str:
    """Keep long names on one line of the drop box."""
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def default_songs_dir() -> Path:
    return Path.home() / "Documents" / "Clone Hero" / "Songs"


class App(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.title("Clone Hero Converter")
        self.geometry("820x640")
        self.minsize(720, 580)
        self.cfg = load_config()
        self.songs_dir = Path(self.cfg.get("songs_dir") or default_songs_dir())
        self.src: Path | None = None
        self.src_url: str | None = None   # a YouTube pick or pasted link, downloaded on "Make my song"
        self.details = None               # search.Details for song.ini, when known
        self.stop = threading.Event()
        self.result: pipeline.Result | None = None
        self.playing: str | None = None
        self.play_buttons: dict = {}
        self.view: ctk.CTkFrame | None = None
        self.q: queue.Queue = queue.Queue()

        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", self.on_drop)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_home()
        self.drain()

    # ---------- view switching ----------
    def switch(self, frame: ctk.CTkFrame):
        if self.view is not None:
            self.view.destroy()
        self.view = frame
        frame.pack(fill="both", expand=True, padx=28, pady=24)

    def header(self, parent, title, subtitle=""):
        ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(size=28, weight="bold")).pack(anchor="w")
        if subtitle:
            ctk.CTkLabel(parent, text=subtitle, text_color=MUTED, font=ctk.CTkFont(size=14),
                         wraplength=720, justify="left").pack(anchor="w", pady=(2, 14))

    # ---------- home ----------
    def show_home(self):
        self.stop_audio()
        f = ctk.CTkFrame(self, fg_color="transparent")
        self.header(f, "Clone Hero Converter",
                    "Drop in a song and get a ready-to-play Clone Hero chart, split into "
                    "guitar, bass, drums and vocals.")
        ctk.CTkButton(f, text="⚙  Settings", width=110, height=32, fg_color=CARD,
                      hover_color=CARD_HOVER, text_color=("#222", "#eee"),
                      command=self.show_settings).place(relx=1.0, y=4, anchor="ne")
        self.drop = ctk.CTkFrame(f, fg_color=CARD, corner_radius=18, border_width=2,
                                 border_color=ACCENT, height=190)
        self.drop.pack(fill="x")
        self.drop.pack_propagate(False)
        self.drop_title = ctk.CTkLabel(self.drop, text="Drop an MP3 or WAV file here",
                                       font=ctk.CTkFont(size=22, weight="bold"))
        self.drop_title.pack(pady=(52, 2))
        self.drop_sub = ctk.CTkLabel(self.drop, text="or click to choose a file", text_color=MUTED)
        self.drop_sub.pack()
        for w in (self.drop, self.drop_title, self.drop_sub):
            w.bind("<Button-1>", lambda e: self.browse())
            w.configure(cursor="hand2")
        self.drop.drop_target_register(DND_FILES)
        self.drop.dnd_bind("<<Drop>>", self.on_drop)

        find = ctk.CTkFrame(f, fg_color="transparent")
        find.pack(fill="x", pady=(12, 0))
        self.query = ctk.CTkEntry(find, height=38,
                                  placeholder_text="...or search for a song (Artist - Title), or paste a YouTube link")
        self.query.pack(side="left", fill="x", expand=True)
        self.query.bind("<Return>", lambda e: self.find())
        ctk.CTkButton(find, text="Search", width=96, height=38, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      command=self.find).pack(side="left", padx=(10, 0))

        form = ctk.CTkFrame(f, fg_color="transparent")
        form.pack(fill="x", pady=(18, 0))
        form.columnconfigure((0, 1), weight=1, uniform="c")
        ctk.CTkLabel(form, text="Artist", text_color=MUTED).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(form, text="Song title", text_color=MUTED).grid(row=0, column=1, sticky="w", padx=(12, 0))
        self.artist = ctk.CTkEntry(form, height=38, placeholder_text="Artist")
        self.title_e = ctk.CTkEntry(form, height=38, placeholder_text="Song title")
        self.artist.grid(row=1, column=0, sticky="ew")
        self.title_e.grid(row=1, column=1, sticky="ew", padx=(12, 0))

        folder = ctk.CTkFrame(f, fg_color="transparent")
        folder.pack(fill="x", pady=(14, 0))
        ctk.CTkLabel(folder, text="Save to", text_color=MUTED).pack(side="left")
        self.folder_lbl = ctk.CTkLabel(folder, text=str(self.songs_dir), anchor="w")
        self.folder_lbl.pack(side="left", padx=10, fill="x", expand=True)
        ctk.CTkButton(folder, text="Change", width=80, fg_color=CARD, hover_color=CARD_HOVER,
                      text_color=("#222", "#eee"), command=self.pick_folder).pack(side="right")

        self.go = ctk.CTkButton(f, text="Make my song", height=48, corner_radius=12,
                                font=ctk.CTkFont(size=17, weight="bold"), fg_color=ACCENT,
                                hover_color=ACCENT_HOVER, state="disabled", command=self.start)
        self.go.pack(fill="x", pady=(22, 0))
        self.hint = ctk.CTkLabel(f, text="Takes a few minutes. A graphics card makes it much faster.",
                                 text_color=MUTED)
        self.hint.pack(pady=(10, 0))
        self.switch(f)
        if self.src:
            self.set_source(self.src)
        elif self.src_url:
            self.set_url(self.src_url, getattr(self, "src_label", "YouTube link"), *self.src_names)

    def browse(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(title="Choose a song",
                                       filetypes=[("Audio", "*.mp3 *.wav *.flac *.ogg *.m4a *.opus"),
                                                  ("All files", "*.*")])
        if p:
            self.set_source(Path(p))

    def on_drop(self, event):
        files = [Path(p) for p in self.tk.splitlist(event.data)]
        files = [p for p in files if p.suffix.lower() in AUDIO_EXT]
        if files:
            self.set_source(files[0])

    def set_source(self, path: Path):
        if getattr(self, "working", False) or not hasattr(self, "artist") or not self.artist.winfo_exists():
            return
        self.src, self.src_url, self.details = path, None, None
        artist, title = pipeline.parse_name(path)
        for entry, val in ((self.artist, artist), (self.title_e, title)):
            entry.delete(0, "end")
            entry.insert(0, val)
        self.drop_title.configure(text=short(path.name))
        self.drop_sub.configure(text="Ready. Check the names below, then press Make my song.")
        self.drop.configure(border_color=GOOD)
        self.go.configure(state="normal")

    def set_url(self, url: str, label: str, artist: str = "", title: str = ""):
        """Use a YouTube video as the song: it is downloaded when Make my song is pressed."""
        if getattr(self, "working", False) or not hasattr(self, "artist") or not self.artist.winfo_exists():
            return
        self.src, self.src_url, self.src_label, self.src_names = None, url, label, (artist, title)
        for entry, val in ((self.artist, artist), (self.title_e, title)):
            entry.delete(0, "end")
            entry.insert(0, val)
        self.drop_title.configure(text=short("YouTube: " + label))
        self.drop_sub.configure(text="Ready. The audio is downloaded when you press Make my song.")
        self.drop.configure(border_color=GOOD)
        self.go.configure(state="normal")

    # ---------- search ----------
    def find(self):
        q = self.query.get().strip()
        if not q:
            return
        if search.is_url(q):
            artist, title = self.artist.get().strip(), self.title_e.get().strip()
            self.details = None
            self.set_url(q, q, artist, title)

            def name_it():  # fill the names from the video's title, e.g. "Artist - Song (Official Video)"
                try:
                    vt, channel = search.video_title(q)
                except Exception:
                    return
                a, t = search.names_from_video(vt, channel)
                self.ui(lambda: self.src_url == q and self.set_url(q, vt, artist or a, title or t))
            threading.Thread(target=name_it, daemon=True).start()
            return
        self.show_search(q)

    def show_search(self, q: str):
        artist, title = search.parse_query(q)
        f = ctk.CTkFrame(self, fg_color="transparent")
        self.back_button(f)
        self.header(f, "Find a song", f"Results for “{q}”")
        self.search_status = ctk.CTkLabel(f, text="Searching...", text_color=MUTED, anchor="w")
        self.search_status.pack(fill="x")
        self.results = ctk.CTkScrollableFrame(f, fg_color="transparent")
        self.results.pack(fill="both", expand=True, pady=(6, 0))
        self.switch(f)
        threading.Thread(target=self.search_work, args=(artist, title), daemon=True).start()

    def search_work(self, artist, title):
        charts, details, videos, errors = [], None, [], []
        try:
            charts = search.community_charts(artist, title)
        except Exception as e:
            errors.append(f"chart search ({e.__class__.__name__})")
        try:
            details = search.song_details(artist, title)
        except Exception:
            details = None  # song details are a bonus; quietly skip
        if details and not artist:
            artist = details.artist
        try:
            videos = search.youtube_search(artist, title, expect_seconds=details.seconds if details else 0)
        except Exception as e:
            errors.append(f"YouTube search ({e.__class__.__name__}; try Settings → Update downloader)")
        self.ui(self.show_results, artist, title, charts, details, videos, errors)

    def section(self, text, sub=""):
        ctk.CTkLabel(self.results, text=text, font=ctk.CTkFont(size=16, weight="bold"), anchor="w"
                     ).pack(fill="x", pady=(14, 0))
        if sub:
            ctk.CTkLabel(self.results, text=sub, text_color=MUTED, anchor="w", wraplength=700,
                         justify="left").pack(fill="x", pady=(0, 4))

    def result_row(self, line1, line2, button, command, badge=""):
        row = ctk.CTkFrame(self.results, fg_color=CARD, corner_radius=12)
        row.pack(fill="x", pady=3)
        txt = ctk.CTkFrame(row, fg_color="transparent")
        txt.pack(side="left", fill="x", expand=True, padx=14, pady=8)
        top = ctk.CTkFrame(txt, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(top, text=line1, anchor="w", font=ctk.CTkFont(size=14), wraplength=520,
                     justify="left").pack(side="left")
        if badge:
            ctk.CTkLabel(top, text=f"  {badge}", text_color=GOOD, font=ctk.CTkFont(size=13, weight="bold")
                         ).pack(side="left")
        ctk.CTkLabel(txt, text=line2, anchor="w", text_color=MUTED).pack(fill="x")
        b = ctk.CTkButton(row, text=button, width=130, fg_color=ACCENT, hover_color=ACCENT_HOVER)
        b.configure(command=lambda: command(b))
        b.pack(side="right", padx=12)
        return b

    def show_results(self, artist, title, charts, details, videos, errors):
        if not getattr(self, "results", None) or not self.results.winfo_exists():
            return  # the user went back
        status = []
        if details:
            bits = [x for x in (details.album, details.year, details.genre) if x]
            status.append(f"{details.artist} – {details.title}" + (f"   ({', '.join(bits)})" if bits else ""))
        if errors:
            status.append("Could not finish: " + "; ".join(errors))
        self.search_status.configure(text="\n".join(status) or "Done.",
                                     text_color="#e0a23a" if errors else MUTED)
        self.section("Already charted by the community",
                     "Real charts made by people on Chorus (enchor.us). If one is here, it will play "
                     "better than a generated chart." if charts else
                     "No existing chart found, so making one below is the way to go.")
        for c in charts:
            line2 = f"charted by {c.charter}" + (f"   ·   {c.parts}" if c.parts else "")
            self.result_row(f"{c.artist} – {c.name}", line2, "Download chart",
                            lambda b, c=c: self.get_chart(b, c))
        self.section("Make a chart from YouTube",
                     "Pick the version to chart. The plain album recording works best; "
                     "“Topic” uploads usually are exactly that.")
        if not videos:
            ctk.CTkLabel(self.results, text="No videos found.", text_color=MUTED, anchor="w").pack(fill="x")
        for v in videos:
            mins = f"{int(v.seconds // 60)}:{int(v.seconds % 60):02d}" if v.seconds else "?"
            self.result_row(v.title, f"{v.channel}   ·   {mins}", "Use this",
                            lambda b, v=v: self.use_video(v, artist, title, details),
                            badge="Best match" if v.best else "")

    def get_chart(self, button, chart):
        button.configure(text="Downloading...", state="disabled")

        def run():
            try:
                path = search.download_chart(chart, self.songs_dir)
                self.ui(lambda: (button.configure(text="Saved ✓"),
                                 self.search_status.configure(
                                     text=f"Saved {path.name} to your songs folder. Rescan songs in "
                                          "Clone Hero to see it.", text_color=GOOD)))
            except Exception as e:
                self.ui(lambda: (button.configure(text="Download chart", state="normal"),
                                 self.search_status.configure(text=f"Download failed: {e}",
                                                              text_color="#e0a23a")))
        threading.Thread(target=run, daemon=True).start()

    def use_video(self, video, artist, title, details):
        self.details = details
        artist = (details.artist if details else "") or artist
        title = (details.title if details else "") or title
        self.show_home()
        self.set_url(video.url, f"{video.title} ({video.channel})", artist, title)

    def pick_folder(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Where should songs be saved?",
                                    initialdir=str(self.songs_dir) if self.songs_dir.exists() else None)
        if d:
            self.songs_dir = Path(d)
            self.folder_lbl.configure(text=d)
            self.cfg["songs_dir"] = d
            save_config(self.cfg)

    # ---------- settings ----------
    def back_button(self, parent, command=None):
        ctk.CTkButton(parent, text="←  Back", width=90, height=32, fg_color=CARD,
                      hover_color=CARD_HOVER, text_color=("#222", "#eee"),
                      command=command or self.show_home).pack(anchor="w", pady=(0, 10))

    def show_settings(self):
        f = ctk.CTkScrollableFrame(self, fg_color="transparent")  # more cards than fit the window
        self.back_button(f)
        self.header(f, "Settings")

        card = ctk.CTkFrame(f, fg_color=CARD, corner_radius=14)
        card.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(card, text="Songs folder", font=ctk.CTkFont(size=15, weight="bold")).pack(
            anchor="w", padx=16, pady=(12, 0))
        ctk.CTkLabel(card, text="Where new songs are saved, and which folder gets tidied.",
                     text_color=MUTED).pack(anchor="w", padx=16)
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(6, 14))
        lbl = ctk.CTkLabel(row, text=str(self.songs_dir), anchor="w")
        lbl.pack(side="left", fill="x", expand=True)

        def change():
            from tkinter import filedialog
            d = filedialog.askdirectory(title="Songs folder",
                                        initialdir=str(self.songs_dir) if self.songs_dir.exists() else None)
            if d:
                self.songs_dir = Path(d)
                lbl.configure(text=d)
                self.cfg["songs_dir"] = d
                save_config(self.cfg)
        ctk.CTkButton(row, text="Change", width=80, command=change, fg_color=ACCENT,
                      hover_color=ACCENT_HOVER).pack(side="right")

        card2 = ctk.CTkFrame(f, fg_color=CARD, corner_radius=14)
        card2.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(card2, text="Charter name", font=ctk.CTkFont(size=15, weight="bold")).pack(
            anchor="w", padx=16, pady=(12, 0))
        ctk.CTkLabel(card2, text="Written into song.ini for songs you make.",
                     text_color=MUTED).pack(anchor="w", padx=16)
        charter = ctk.CTkEntry(card2, height=36)
        charter.insert(0, self.cfg.get("charter", ""))
        charter.pack(fill="x", padx=16, pady=(6, 14))

        def save_charter(_e=None):
            self.cfg["charter"] = charter.get().strip()
            save_config(self.cfg)
        charter.bind("<FocusOut>", save_charter)
        charter.bind("<Return>", save_charter)

        card3 = ctk.CTkFrame(f, fg_color=CARD, corner_radius=14)
        card3.pack(fill="x")
        ctk.CTkLabel(card3, text="Tidy song names", font=ctk.CTkFont(size=15, weight="bold")).pack(
            anchor="w", padx=16, pady=(12, 0))
        ctk.CTkLabel(card3, text="Scan your songs folder and clean up names like "
                                 "\u201cArtist - Song (Live) (Harmonix)\u201d \u2192 \u201cArtist - Song\u201d. "
                                 "You review every change first, and it can be undone.",
                     text_color=MUTED, wraplength=680, justify="left").pack(anchor="w", padx=16)
        ctk.CTkButton(card3, text="Scan songs folder...", height=38, fg_color=ACCENT,
                      hover_color=ACCENT_HOVER, command=lambda: (save_charter(), self.show_tidy())
                      ).pack(anchor="w", padx=16, pady=(8, 14))

        card4 = ctk.CTkFrame(f, fg_color=CARD, corner_radius=14)
        card4.pack(fill="x", pady=(12, 0))
        ctk.CTkLabel(card4, text="YouTube downloader", font=ctk.CTkFont(size=15, weight="bold")).pack(
            anchor="w", padx=16, pady=(12, 0))
        ctk.CTkLabel(card4, text="YouTube changes often and older downloader versions stop working. "
                                 "If searching or downloading fails, update it here.",
                     text_color=MUTED, wraplength=680, justify="left").pack(anchor="w", padx=16)
        row4 = ctk.CTkFrame(card4, fg_color="transparent")
        row4.pack(fill="x", padx=16, pady=(8, 14))
        upd_msg = ctk.CTkLabel(row4, text="", text_color=MUTED, anchor="w")

        def update():
            upd.configure(state="disabled", text="Updating...")

            def run():
                msg = search.update_downloader()
                self.ui(lambda: upd_msg.winfo_exists() and (
                    upd.configure(state="normal", text="Update downloader"),
                    upd_msg.configure(text=msg, text_color=GOOD if "up to date" in msg else "#e0a23a")))
            threading.Thread(target=run, daemon=True).start()
        upd = ctk.CTkButton(row4, text="Update downloader", height=38, fg_color=ACCENT,
                            hover_color=ACCENT_HOVER, command=update)
        upd.pack(side="left")
        upd_msg.pack(side="left", padx=12, fill="x", expand=True)
        self.switch(f)

    # ---------- tidy names ----------
    def show_tidy(self, message=""):
        f = ctk.CTkFrame(self, fg_color="transparent")
        self.back_button(f, self.show_settings)
        self.header(f, "Tidy song names", str(self.songs_dir))
        self.tidy_parens = ctk.BooleanVar(value=self.cfg.get("tidy_parens", True))
        self.tidy_brackets = ctk.BooleanVar(value=self.cfg.get("tidy_brackets", True))
        opts = ctk.CTkFrame(f, fg_color="transparent")
        opts.pack(fill="x")
        ctk.CTkCheckBox(opts, text="Remove (parentheses)", variable=self.tidy_parens,
                        command=self.rescan).pack(side="left")
        ctk.CTkCheckBox(opts, text="Remove [brackets]", variable=self.tidy_brackets,
                        command=self.rescan).pack(side="left", padx=18)
        self.tidy_msg = ctk.CTkLabel(f, text=message, text_color=GOOD, anchor="w")
        self.tidy_msg.pack(fill="x", pady=(8, 0))
        self.tidy_list = ctk.CTkScrollableFrame(f, fg_color=CARD, corner_radius=12, height=300)
        self.tidy_list.pack(fill="both", expand=True, pady=8)
        bar = ctk.CTkFrame(f, fg_color="transparent")
        bar.pack(fill="x")
        self.tidy_apply = ctk.CTkButton(bar, text="Rename selected", height=42, fg_color=ACCENT,
                                        hover_color=ACCENT_HOVER, command=self.apply_tidy)
        self.tidy_apply.pack(side="left", expand=True, fill="x")
        self.tidy_undo = ctk.CTkButton(bar, text="Undo last rename", height=42, fg_color=CARD,
                                       hover_color=CARD_HOVER, text_color=("#222", "#eee"),
                                       command=self.undo_tidy)
        self.tidy_undo.pack(side="left", expand=True, fill="x", padx=(12, 0))
        self.switch(f)
        self.rescan()

    def rescan(self):
        self.cfg["tidy_parens"] = self.tidy_parens.get()
        self.cfg["tidy_brackets"] = self.tidy_brackets.get()
        save_config(self.cfg)
        for w in self.tidy_list.winfo_children():
            w.destroy()
        self.proposals = []
        if not self.songs_dir.exists():
            ctk.CTkLabel(self.tidy_list, text="The songs folder does not exist yet.",
                         text_color=MUTED).pack(pady=20)
        else:
            self.proposals = tidy.scan(self.songs_dir, self.tidy_parens.get(), self.tidy_brackets.get())
        self.tidy_vars = []
        for p in self.proposals:
            var = ctk.BooleanVar(value=p.selected)
            self.tidy_vars.append(var)
            row = ctk.CTkFrame(self.tidy_list, fg_color="transparent")
            row.pack(fill="x", pady=3, padx=6)
            ctk.CTkCheckBox(row, text="", variable=var, width=24,
                            state="normal" if p.ok else "disabled").pack(side="left")
            txt = ctk.CTkFrame(row, fg_color="transparent")
            txt.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(txt, text=p.path.name, anchor="w", text_color=MUTED, wraplength=640,
                         justify="left").pack(fill="x")
            ctk.CTkLabel(txt, text="\u2192  " + p.new_name + (f"   (skipped: {p.note})" if p.note else ""),
                         anchor="w", wraplength=640, justify="left",
                         text_color=("#111", "#fff") if p.ok else "#e0a23a").pack(fill="x")
        if self.songs_dir.exists() and not self.proposals:
            ctk.CTkLabel(self.tidy_list, text="Nothing to tidy. All song names are already clean.",
                         text_color=MUTED).pack(pady=24)
        self.tidy_apply.configure(state="normal" if any(p.ok for p in self.proposals) else "disabled")
        self.tidy_undo.configure(state="normal" if tidy.can_undo(self.songs_dir) else "disabled")

    def apply_tidy(self):
        for p, v in zip(self.proposals, self.tidy_vars):
            p.selected = v.get()
        n, errs = tidy.apply(self.proposals, self.songs_dir, self.tidy_parens.get(),
                             self.tidy_brackets.get())
        msg = f"Renamed {n} folder{'s' if n != 1 else ''}." + (f" {len(errs)} failed: {errs[0]}" if errs else "")
        self.show_tidy(msg)

    def undo_tidy(self):
        n, errs = tidy.undo(self.songs_dir)
        self.show_tidy(f"Restored {n} folder name{'s' if n != 1 else ''}." + (f" {errs[0]}" if errs else ""))

    # ---------- progress ----------
    def start(self):
        if not (self.src or self.src_url):
            return
        self.cfg["songs_dir"] = str(self.songs_dir)
        save_config(self.cfg)
        fallback = self.src.stem if self.src else "Untitled"
        artist, title = self.artist.get().strip(), self.title_e.get().strip() or fallback
        self.working = True
        self.stop.clear()
        # a YouTube source gets an extra first step; the pipeline's steps follow it
        self.steps = (["Downloading the song"] if self.src_url else []) + pipeline.STEPS

        f = ctk.CTkFrame(self, fg_color="transparent")
        self.header(f, "Making your song", f"{artist + ' - ' if artist else ''}{title}")
        card = ctk.CTkFrame(f, fg_color=CARD, corner_radius=18)
        card.pack(fill="x")
        self.step_lbls = []
        for s in self.steps:
            lbl = ctk.CTkLabel(card, text=f"○  {s}", anchor="w", font=ctk.CTkFont(size=16),
                               text_color=MUTED)
            lbl.pack(fill="x", padx=24, pady=(14 if not self.step_lbls else 6, 0))
            self.step_lbls.append(lbl)
        ctk.CTkLabel(card, text="").pack(pady=4)
        self.bar = ctk.CTkProgressBar(f, height=14, progress_color=ACCENT)
        self.bar.set(0)
        self.bar.pack(fill="x", pady=(22, 6))
        self.pct = ctk.CTkLabel(f, text="Starting...", text_color=MUTED)
        self.pct.pack()
        ctk.CTkButton(f, text="Cancel", width=110, fg_color=CARD, hover_color=CARD_HOVER,
                      text_color=("#222", "#eee"), command=self.cancel).pack(pady=18)
        self.switch(f)

        threading.Thread(target=self.work, args=(self.src, self.src_url, artist, title, self.details),
                         daemon=True).start()

    def cancel(self):
        self.stop.set()
        self.pct.configure(text="Cancelling...")

    def ui(self, fn, *a):
        """Run fn(*a) on the main thread (safe to call from the worker thread)."""
        self.q.put((fn, a))

    def drain(self):
        try:
            while True:
                fn, a = self.q.get_nowait()
                fn(*a)
        except queue.Empty:
            pass
        self.after(50, self.drain)

    def set_step(self, i):
        for n, lbl in enumerate(self.step_lbls):
            name = self.steps[n]
            if n < i:
                lbl.configure(text=f"✓  {name}", text_color=GOOD)
            elif n == i:
                lbl.configure(text=f"●  {name}...", text_color=("#111", "#fff"))
            else:
                lbl.configure(text=f"○  {name}", text_color=MUTED)

    def set_progress(self, frac):
        self.bar.set(frac)
        self.pct.configure(text=f"{int(frac * 100)}%")

    def work(self, src, url, artist, title, details):
        tmp = Path(tempfile.mkdtemp(prefix="chc_get_"))
        try:
            offset, start = 0, 0.0
            if url:
                self.ui(self.set_step, 0)
                src = search.youtube_download(url, tmp, lambda fr: self.ui(self.set_progress, 0.12 * fr), self.stop)
                offset, start = 1, 0.12
            # album, year, genre and cover for song.ini / album.jpg; skipped quietly when offline
            cover = None
            try:
                details = details or search.song_details(artist, title)
                cover = search.cover_art(details, tmp / "cover.jpg") if details else None
            except Exception:
                details = None
            res = pipeline.make_song(
                src, artist, title, self.songs_dir, charter=self.cfg.get("charter", ""),
                on_step=lambda i: self.ui(self.set_step, i + offset),
                on_progress=lambda fr: self.ui(self.set_progress, start + (1 - start) * fr), stop=self.stop,
                details=details, cover=cover)
            self.ui(self.show_result, res)
        except pipeline.Cancelled:
            self.ui(self.finish_busy)
            self.ui(self.show_home)
        except Exception as e:
            if self.stop.is_set():  # cancelled during the download
                self.ui(self.finish_busy)
                self.ui(self.show_home)
                return
            traceback.print_exc()
            self.ui(self.finish_busy)
            msg = str(e)
            if url and "youtube" in traceback.format_exc().lower():
                msg += "\n\nIf YouTube downloads keep failing, try Settings → Update downloader."
            self.ui(self.show_error, msg)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def finish_busy(self):
        self.working = False

    # ---------- result ----------
    def show_error(self, msg):
        f = ctk.CTkFrame(self, fg_color="transparent")
        self.header(f, "Something went wrong", "Your song could not be made. Details below.")
        box = ctk.CTkTextbox(f, height=240)
        box.insert("1.0", msg)
        box.configure(state="disabled")
        box.pack(fill="x")
        ctk.CTkButton(f, text="Try again", height=42, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      command=self.show_home).pack(pady=18)
        self.switch(f)

    def show_result(self, res: pipeline.Result):
        self.working = False
        self.result = res
        f = ctk.CTkFrame(self, fg_color="transparent")
        self.header(f, "Your song is ready ✓", f"Saved to {res.song_dir}")

        ctk.CTkLabel(f, text="Listen to the instruments", font=ctk.CTkFont(size=15, weight="bold")
                     ).pack(anchor="w", pady=(0, 6))
        self.play_buttons = {}
        for label, path in res.stems.items():
            row = ctk.CTkFrame(f, fg_color=CARD, corner_radius=12)
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=15)).pack(side="left", padx=16, pady=10)
            if winsound:
                b = ctk.CTkButton(row, text="▶  Play", width=96, fg_color=ACCENT,
                                  hover_color=ACCENT_HOVER,
                                  command=lambda l=label, p=path: self.toggle_play(l, p))
                b.pack(side="right", padx=12, pady=8)
                self.play_buttons[label] = b

        names = {"PART GUITAR": "Guitar", "PART BASS": "Bass", "PART DRUMS": "Drums"}
        summary = "   ·   ".join(f"{names[k]}: {v} notes" for k, v in res.counts.items())
        info = ctk.CTkFrame(f, fg_color=CARD, corner_radius=12)
        info.pack(fill="x", pady=(14, 0))
        ctk.CTkLabel(info, text=f"Detected tempo ~{res.bpm:g} BPM   ·   {summary}",
                     wraplength=700, justify="left").pack(anchor="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(info, text="The chart is an automatic first draft on all four difficulties. "
                                "To fine-tune it, open notes.mid in Moonscraper.",
                     text_color=MUTED, wraplength=700, justify="left").pack(anchor="w", padx=16, pady=(0, 12))

        btns = ctk.CTkFrame(f, fg_color="transparent")
        btns.pack(fill="x", pady=(18, 0))
        ctk.CTkButton(btns, text="Open song folder", height=44, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      command=lambda: os.startfile(res.song_dir)).pack(side="left", expand=True, fill="x")
        ctk.CTkButton(btns, text="Make another song", height=44, fg_color=CARD, hover_color=CARD_HOVER,
                      text_color=("#222", "#eee"), command=self.another).pack(
            side="left", expand=True, fill="x", padx=(12, 0))
        self.switch(f)

    def another(self):
        self.stop_audio()
        self.cleanup_preview()
        self.src, self.src_url, self.details = None, None, None
        self.show_home()

    # ---------- audio preview ----------
    def stop_audio(self):
        if winsound:
            winsound.PlaySound(None, winsound.SND_PURGE)
        for lbl, b in self.play_buttons.items():
            try:
                b.configure(text="▶  Play")
            except Exception:
                pass
        self.playing = None

    def toggle_play(self, label, path):
        was = self.playing
        self.stop_audio()
        if was == label:
            return
        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        self.playing = label
        self.play_buttons[label].configure(text="■  Stop")

    def cleanup_preview(self):
        if self.result:
            shutil.rmtree(self.result.preview_dir, ignore_errors=True)
            self.result = None

    def on_close(self):
        self.stop.set()
        self.stop_audio()
        self.cleanup_preview()
        self.destroy()


if __name__ == "__main__":
    if getattr(sys, "frozen", False) is False:
        os.chdir(Path(__file__).resolve().parent)
    App().mainloop()
