#!/usr/bin/env python
"""Clone Hero Converter: drop in a song, get a Clone Hero song folder."""
import json
import os
import queue
import shutil
import sys
import threading
import traceback
from pathlib import Path

import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD

from chc import pipeline, tidy

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
        self.src = path
        artist, title = pipeline.parse_name(path)
        for entry, val in ((self.artist, artist), (self.title_e, title)):
            entry.delete(0, "end")
            entry.insert(0, val)
        self.drop_title.configure(text=path.name)
        self.drop_sub.configure(text="Ready. Check the names below, then press Make my song.")
        self.drop.configure(border_color=GOOD)
        self.go.configure(state="normal")

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
        f = ctk.CTkFrame(self, fg_color="transparent")
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
        if not self.src:
            return
        self.cfg["songs_dir"] = str(self.songs_dir)
        save_config(self.cfg)
        artist, title = self.artist.get().strip(), self.title_e.get().strip() or self.src.stem
        self.working = True
        self.stop.clear()

        f = ctk.CTkFrame(self, fg_color="transparent")
        self.header(f, "Making your song", f"{artist + ' - ' if artist else ''}{title}")
        card = ctk.CTkFrame(f, fg_color=CARD, corner_radius=18)
        card.pack(fill="x")
        self.step_lbls = []
        for s in pipeline.STEPS:
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

        threading.Thread(target=self.work, args=(self.src, artist, title), daemon=True).start()

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
            name = pipeline.STEPS[n]
            if n < i:
                lbl.configure(text=f"✓  {name}", text_color=GOOD)
            elif n == i:
                lbl.configure(text=f"●  {name}...", text_color=("#111", "#fff"))
            else:
                lbl.configure(text=f"○  {name}", text_color=MUTED)

    def set_progress(self, frac):
        self.bar.set(frac)
        self.pct.configure(text=f"{int(frac * 100)}%")

    def work(self, src, artist, title):
        try:
            res = pipeline.make_song(
                src, artist, title, self.songs_dir, charter=self.cfg.get("charter", ""),
                on_step=lambda i: self.ui(self.set_step, i),
                on_progress=lambda fr: self.ui(self.set_progress, fr), stop=self.stop)
            self.ui(self.show_result, res)
        except pipeline.Cancelled:
            self.ui(self.finish_busy)
            self.ui(self.show_home)
        except Exception as e:
            traceback.print_exc()
            self.ui(self.finish_busy)
            self.ui(self.show_error, str(e))

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
        self.src = None
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
