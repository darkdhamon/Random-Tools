from __future__ import annotations

import csv
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .models import ensure_models
from .scanner import FaceEngine, MatchResult, ScanProgress, build_reference_embeddings, scan_folder


class FaceFinderApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Face Photo Finder")
        self.geometry("980x650")
        self.minsize(760, 500)
        self.references: list[Path] = []
        self.matches: list[MatchResult] = []
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.folder_var = tk.StringVar()
        self.threshold_var = tk.DoubleVar(value=0.45)
        self.status_var = tk.StringVar(value="Choose reference photos and a folder to scan.")
        self._build()
        self.after(100, self._drain_events)

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Reference photos (one clear face per photo)").grid(row=0, column=0, sticky="w")
        self.reference_label = ttk.Label(outer, text="None selected")
        self.reference_label.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(outer, text="Choose photos…", command=self.choose_references).grid(row=1, column=1)

        ttk.Label(outer, text="Folder to scan (subfolders included)").grid(row=2, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(outer, textvariable=self.folder_var).grid(row=3, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(outer, text="Choose folder…", command=self.choose_folder).grid(row=3, column=1)

        controls = ttk.Frame(outer)
        controls.grid(row=4, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(controls, text="Match strictness").pack(side="left")
        ttk.Scale(controls, from_=0.30, to=0.70, variable=self.threshold_var, length=220).pack(side="left", padx=8)
        self.threshold_label = ttk.Label(controls, width=5)
        self.threshold_label.pack(side="left")
        self.threshold_var.trace_add("write", lambda *_: self.threshold_label.config(text=f"{self.threshold_var.get():.2f}"))
        self.threshold_label.config(text="0.45")
        self.scan_button = ttk.Button(controls, text="Start scan", command=self.start_scan)
        self.scan_button.pack(side="right")
        self.cancel_button = ttk.Button(controls, text="Cancel", command=self.cancel_scan, state="disabled")
        self.cancel_button.pack(side="right", padx=8)

        self.progress = ttk.Progressbar(outer, mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=2, sticky="ew")
        ttk.Label(outer, textvariable=self.status_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 8))

        self.tree = ttk.Treeview(outer, columns=("score", "faces", "path"), show="headings", selectmode="extended")
        self.tree.heading("score", text="Similarity")
        self.tree.heading("faces", text="Faces")
        self.tree.heading("path", text="Photo")
        self.tree.column("score", width=90, anchor="center", stretch=False)
        self.tree.column("faces", width=60, anchor="center", stretch=False)
        self.tree.column("path", width=700)
        self.tree.grid(row=7, column=0, columnspan=2, sticky="nsew")
        self.tree.bind("<Double-1>", lambda _event: self.open_selected())

        actions = ttk.Frame(outer)
        actions.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(actions, text="Open selected", command=self.open_selected).pack(side="left")
        ttk.Button(actions, text="Export CSV…", command=self.export_csv).pack(side="left", padx=8)
        ttk.Button(actions, text="Copy matches…", command=self.copy_matches).pack(side="left")
        ttk.Label(actions, text="Verify matches before relying on them; face recognition can be wrong.").pack(side="right")

        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(7, weight=1)

    def choose_references(self) -> None:
        names = filedialog.askopenfilenames(title="Choose reference photos", filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff")])
        if names:
            self.references = [Path(name) for name in names]
            self.reference_label.config(text=f"{len(self.references)} selected: " + ", ".join(path.name for path in self.references[:3]))

    def choose_folder(self) -> None:
        name = filedialog.askdirectory(title="Choose folder to scan")
        if name:
            self.folder_var.set(name)

    def start_scan(self) -> None:
        folder = Path(self.folder_var.get())
        if not self.references:
            messagebox.showerror("Reference required", "Choose at least one reference photo.")
            return
        if not folder.is_dir():
            messagebox.showerror("Folder required", "Choose an existing folder to scan.")
            return
        self.matches.clear()
        self.tree.delete(*self.tree.get_children())
        self.cancel_event.clear()
        self.scan_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.progress["value"] = 0
        threshold = self.threshold_var.get()
        self.worker = threading.Thread(target=self._scan_worker, args=(folder, threshold), daemon=True)
        self.worker.start()

    def _scan_worker(self, folder: Path, threshold: float) -> None:
        try:
            model_dir = Path(__file__).resolve().parents[1] / "models"
            detector, recognizer = ensure_models(model_dir, lambda value: self.events.put(("status", value)))
            engine = FaceEngine(detector, recognizer)
            self.events.put(("status", "Reading reference faces…"))
            references = build_reference_embeddings(engine, self.references)
            results = scan_folder(engine, folder, references, threshold, self._on_progress, self.cancel_event)
            self.events.put(("done", results))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _on_progress(self, value: ScanProgress) -> None:
        self.events.put(("progress", value))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "progress":
                    value = payload
                    assert isinstance(value, ScanProgress)
                    self.progress["maximum"] = max(value.total, 1)
                    self.progress["value"] = value.completed
                    self.status_var.set(f"Scanning {value.completed}/{value.total}: {value.path.name}")
                    if value.match:
                        self._add_match(value.match)
                elif kind == "done":
                    self.matches = list(payload)  # type: ignore[arg-type]
                    self._finish(f"Found {len(self.matches)} matching photo(s)." if not self.cancel_event.is_set() else f"Cancelled. Found {len(self.matches)} match(es).")
                elif kind == "error":
                    self._finish("Scan failed.")
                    messagebox.showerror("Scan failed", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _add_match(self, match: MatchResult) -> None:
        self.tree.insert("", "end", iid=str(match.path), values=(f"{match.score:.3f}", match.face_count, str(match.path)))

    def _finish(self, status: str) -> None:
        self.status_var.set(status)
        self.scan_button.config(state="normal")
        self.cancel_button.config(state="disabled")

    def cancel_scan(self) -> None:
        self.cancel_event.set()
        self.status_var.set("Cancelling after the current photo…")

    def selected_paths(self) -> list[Path]:
        selected = self.tree.selection() or self.tree.get_children()
        return [Path(self.tree.item(item, "values")[2]) for item in selected]

    def open_selected(self) -> None:
        paths = self.selected_paths()
        if paths:
            if sys.platform == "win32":
                os.startfile(paths[0])  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(paths[0])])
            else:
                subprocess.Popen(["xdg-open", str(paths[0])])

    def export_csv(self) -> None:
        if not self.matches:
            messagebox.showinfo("Nothing to export", "Run a scan that finds matches first.")
            return
        name = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if name:
            with Path(name).open("w", newline="", encoding="utf-8-sig") as output:
                writer = csv.writer(output)
                writer.writerow(["path", "similarity", "faces_detected"])
                writer.writerows((str(item.path), f"{item.score:.6f}", item.face_count) for item in self.matches)
            self.status_var.set(f"Exported {len(self.matches)} rows to {name}")

    def copy_matches(self) -> None:
        paths = self.selected_paths()
        if not paths:
            messagebox.showinfo("Nothing to copy", "Select matches or run a scan first.")
            return
        target_name = filedialog.askdirectory(title="Choose destination for copies")
        if not target_name:
            return
        target = Path(target_name).resolve()
        copied = 0
        for source in paths:
            destination = target / source.name
            number = 2
            while destination.exists():
                destination = target / f"{source.stem}_{number}{source.suffix}"
                number += 1
            shutil.copy2(source, destination)
            copied += 1
        self.status_var.set(f"Copied {copied} photo(s) to {target}")


def main() -> None:
    FaceFinderApp().mainloop()


if __name__ == "__main__":
    main()
