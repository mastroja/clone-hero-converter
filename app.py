#!/usr/bin/env python
"""Clone Hero Converter: desktop GUI for stem separation and MIDI conversion."""
import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import midi_to_ch
import separate

CONFIG = Path.home() / ".clone_hero_converter.json"


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


class LogBox(ttk.Frame):
    """Read-only log; lines flagged progress replace the previous progress line."""

    def __init__(self, master):
        super().__init__(master)
        self.text = tk.Text(self, height=12, state="disabled", wrap="word")
        sb = ttk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        self.text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._last_progress = False

    def write(self, msg, progress=False):
        t = self.text
        t.configure(state="normal")
        if self._last_progress:
            t.delete("end-2l", "end-1l")
        t.insert("end", msg + "\n")
        t.see("end")
        t.configure(state="disabled")
        self._last_progress = progress


def path_row(parent, row, label, var, browse):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
    ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6)
    ttk.Button(parent, text="Browse...", command=browse).grid(row=row, column=2)


class SeparateTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=10)
        self.app = app
        self.files: list[Path] = []
        self.stop = threading.Event()
        self.q: queue.Queue = queue.Queue()
        self.columnconfigure(1, weight=1)

        self.out = tk.StringVar(value=app.cfg.get("songs_dir", ""))
        self.model = tk.StringVar(value=app.cfg.get("model", "htdemucs_ft"))
        self.device = tk.StringVar(value=app.cfg.get("device", "cuda"))
        self.charter = tk.StringVar(value=app.cfg.get("charter", ""))
        self.force = tk.BooleanVar()

        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Button(bar, text="Add files...", command=self.add_files).pack(side="left")
        ttk.Button(bar, text="Add folder...", command=self.add_folder).pack(side="left", padx=6)
        ttk.Button(bar, text="Remove selected", command=self.remove).pack(side="left")
        ttk.Button(bar, text="Clear", command=self.clear).pack(side="left", padx=6)

        self.listbox = tk.Listbox(self, height=7, selectmode="extended")
        self.listbox.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=6)
        self.rowconfigure(1, weight=1)

        path_row(self, 2, "Songs folder:", self.out, self.pick_out)
        opts = ttk.Frame(self)
        opts.grid(row=3, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Label(opts, text="Model:").pack(side="left")
        ttk.Combobox(opts, textvariable=self.model, width=14,
                     values=["htdemucs_ft", "htdemucs", "htdemucs_6s", "mdx_extra"]).pack(side="left", padx=4)
        ttk.Label(opts, text="Device:").pack(side="left", padx=(10, 0))
        ttk.Combobox(opts, textvariable=self.device, width=6, values=["cuda", "cpu"],
                     state="readonly").pack(side="left", padx=4)
        ttk.Label(opts, text="Charter:").pack(side="left", padx=(10, 0))
        ttk.Entry(opts, textvariable=self.charter, width=16).pack(side="left", padx=4)
        ttk.Checkbutton(opts, text="Redo existing", variable=self.force).pack(side="left", padx=10)

        run = ttk.Frame(self)
        run.grid(row=4, column=0, columnspan=3, sticky="ew")
        self.go = ttk.Button(run, text="Separate stems", command=self.start)
        self.go.pack(side="left")
        self.cancel = ttk.Button(run, text="Cancel", command=self.stop.set, state="disabled")
        self.cancel.pack(side="left", padx=6)
        self.status = ttk.Label(run, text="")
        self.status.pack(side="left", padx=10)
        self.bar = ttk.Progressbar(self, mode="determinate")
        self.bar.grid(row=5, column=0, columnspan=3, sticky="ew", pady=6)
        self.log = LogBox(self)
        self.log.grid(row=6, column=0, columnspan=3, sticky="nsew")
        self.rowconfigure(6, weight=2)

    def refresh(self):
        self.listbox.delete(0, "end")
        for f in self.files:
            self.listbox.insert("end", f.name)

    def add_files(self):
        for f in filedialog.askopenfilenames(
                filetypes=[("Audio", "*.mp3 *.wav *.flac *.ogg *.m4a *.opus"), ("All", "*.*")]):
            if Path(f) not in self.files:
                self.files.append(Path(f))
        self.refresh()

    def add_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.files += [f for f in separate.collect_audio(Path(d)) if f not in self.files]
            self.refresh()

    def remove(self):
        for i in reversed(self.listbox.curselection()):
            del self.files[i]
        self.refresh()

    def clear(self):
        self.files.clear()
        self.refresh()

    def pick_out(self):
        d = filedialog.askdirectory()
        if d:
            self.out.set(d)

    def start(self):
        if not self.files:
            return messagebox.showinfo("Nothing to do", "Add at least one audio file.")
        if not self.out.get():
            return messagebox.showinfo("Songs folder", "Choose where song folders should go.")
        self.app.cfg.update(songs_dir=self.out.get(), model=self.model.get(),
                            device=self.device.get(), charter=self.charter.get())
        save_config(self.app.cfg)
        self.stop.clear()
        self.go.configure(state="disabled")
        self.cancel.configure(state="normal")
        self.bar.configure(maximum=len(self.files), value=0)
        files, args = list(self.files), (Path(self.out.get()), self.model.get(),
                                         self.device.get(), self.charter.get(), self.force.get())

        def work():
            fails = separate.run_batch(
                files, *args, log=lambda m, progress=False: self.q.put(("log", m, progress)),
                stop=self.stop, on_item=lambda i, n, name: self.q.put(("item", i, n, name)))
            self.q.put(("done", fails))

        threading.Thread(target=work, daemon=True).start()
        self.after(100, self.poll)

    def poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "log":
                    self.log.write(msg[1], msg[2])
                elif msg[0] == "item":
                    self.bar.configure(value=msg[1])
                    self.status.configure(text=f"{msg[1] + 1 if msg[3] else msg[1]}/{msg[2]} {msg[3]}")
                else:
                    self.go.configure(state="normal")
                    self.cancel.configure(state="disabled")
                    self.status.configure(text=f"Finished, {msg[1]} failed" if msg[1] else "Finished")
                    return
        except queue.Empty:
            pass
        self.after(100, self.poll)


class ConvertTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=10)
        self.app = app
        self.columnconfigure(1, weight=1)
        self.src = tk.StringVar()
        self.dest = tk.StringVar()
        self.inst = tk.StringVar(value="guitar")
        self.track = tk.StringVar(value="(busiest)")
        self.mode = tk.StringVar(value="quantile")
        self.gap = tk.DoubleVar(value=0.125)
        self.sus = tk.BooleanVar()

        path_row(self, 0, "Source MIDI:", self.src, self.pick_src)
        path_row(self, 1, "Output notes.mid:", self.dest, self.pick_dest)
        ttk.Button(self, text="Use a song folder...", command=self.pick_song).grid(
            row=2, column=1, sticky="w", padx=6)

        f = ttk.LabelFrame(self, text="Options", padding=8)
        f.grid(row=3, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(f, text="Instrument:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.inst, values=list(midi_to_ch.TRACK_NAMES),
                     state="readonly", width=10).grid(row=0, column=1, padx=6, pady=3)
        ttk.Label(f, text="Source track:").grid(row=0, column=2, padx=(14, 0))
        self.track_box = ttk.Combobox(f, textvariable=self.track, values=["(busiest)"],
                                      state="readonly", width=34)
        self.track_box.grid(row=0, column=3, padx=6)
        ttk.Label(f, text="Lane mode:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.mode, values=["quantile", "contour", "single"],
                     state="readonly", width=10).grid(row=1, column=1, padx=6, pady=3)
        ttk.Label(f, text="Min gap (beats):").grid(row=1, column=2, padx=(14, 0))
        ttk.Spinbox(f, from_=0, to=2, increment=0.0625, textvariable=self.gap,
                    width=8).grid(row=1, column=3, sticky="w", padx=6)
        ttk.Checkbutton(f, text="Keep sustains (guitar/bass)", variable=self.sus).grid(
            row=2, column=0, columnspan=3, sticky="w")
        ttk.Label(f, text="quantile = spread by pitch, contour = follow up/down movement, "
                          "single = all green", foreground="gray").grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))

        ttk.Button(self, text="Convert", command=self.run).grid(row=4, column=0, sticky="w")
        self.log = LogBox(self)
        self.log.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=8)
        self.rowconfigure(5, weight=1)

    def pick_src(self):
        f = filedialog.askopenfilename(filetypes=[("MIDI", "*.mid *.midi"), ("All", "*.*")])
        if not f:
            return
        self.src.set(f)
        if not self.dest.get():
            self.dest.set(str(Path(f).with_name("notes.mid")))
        try:
            tracks = midi_to_ch.list_tracks(f)
        except Exception as e:  # unreadable MIDI
            return messagebox.showerror("MIDI error", str(e))
        self.track_box.configure(values=["(busiest)"] + [f"{i}: {n or '(unnamed)'} ({c} notes)"
                                                         for i, n, c in tracks])
        self.track.set("(busiest)")

    def pick_dest(self):
        f = filedialog.asksaveasfilename(defaultextension=".mid", initialfile="notes.mid",
                                         filetypes=[("MIDI", "*.mid")])
        if f:
            self.dest.set(f)

    def pick_song(self):
        d = filedialog.askdirectory(initialdir=self.app.cfg.get("songs_dir") or None)
        if d:
            self.dest.set(str(Path(d) / "notes.mid"))

    def run(self):
        if not self.src.get():
            return messagebox.showinfo("Source MIDI", "Choose a source MIDI file.")
        sel = self.track.get()
        track = None if sel == "(busiest)" else sel.split(":")[0]
        try:
            dest, n = midi_to_ch.convert(
                self.src.get(), self.dest.get() or None, self.inst.get(), track,
                self.mode.get(), float(self.gap.get()), self.sus.get())
        except (SystemExit, Exception) as e:
            self.log.write(f"FAILED: {e}")
            return messagebox.showerror("Conversion failed", str(e))
        self.log.write(f"Wrote {dest}\n{n} notes on {midi_to_ch.TRACK_NAMES[self.inst.get()]}. "
                       "Open it in Moonscraper (Load) to set BPM markers and offset.")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clone Hero Converter")
        self.geometry("760x620")
        self.cfg = load_config()
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        nb.add(SeparateTab(nb, self), text="1. Separate stems")
        nb.add(ConvertTab(nb, self), text="2. Convert MIDI")


if __name__ == "__main__":
    App().mainloop()
