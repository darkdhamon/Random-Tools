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

import cv2
from PIL import Image, ImageDraw, ImageOps, ImageTk

from .models import ensure_models
from .scanner import DetectedFace, FaceEngine, MatchResult, ScanProgress, build_reference_embeddings, scan_folder
from .settings import AppSettings, load_settings, save_settings


class FaceSelectionRequest:
    def __init__(self, path: Path, faces: list[DetectedFace]) -> None:
        self.path = path
        self.faces = faces
        self.selected: list[int] = []
        self.ready = threading.Event()


class FaceFinderApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Face Photo Finder")
        self.geometry("1080x780")
        self.minsize(800, 650)
        self.references: list[Path] = []
        self.matches: list[MatchResult] = []
        self.reference_photos: list[ImageTk.PhotoImage] = []
        self.result_photos: dict[str, ImageTk.PhotoImage] = {}
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.face_dialog: tk.Toplevel | None = None
        self.face_request: FaceSelectionRequest | None = None
        self.folder_var = tk.StringVar()
        self.threshold_var = tk.DoubleVar(value=0.45)
        self.status_var = tk.StringVar(value="Choose reference photos and a folder to scan.")
        self._build()
        self._restore_settings()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(100, self._drain_events)

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Reference photos (one clear face per photo)").grid(row=0, column=0, sticky="w")
        reference_panel = ttk.Frame(outer)
        reference_panel.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.reference_label = ttk.Label(reference_panel, text="None selected")
        self.reference_label.pack(anchor="w")
        self.reference_thumbnails = ttk.Frame(reference_panel)
        self.reference_thumbnails.pack(anchor="w", pady=(6, 0))
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

        style = ttk.Style(self)
        style.configure("Results.Treeview", rowheight=84)
        self.tree = ttk.Treeview(
            outer,
            columns=("score", "faces", "path"),
            show="tree headings",
            selectmode="extended",
            style="Results.Treeview",
        )
        self.tree.heading("#0", text="Preview")
        self.tree.heading("score", text="Similarity")
        self.tree.heading("faces", text="Faces")
        self.tree.heading("path", text="Photo")
        self.tree.column("#0", width=120, minwidth=120, anchor="center", stretch=False)
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
            self._show_reference_thumbnails()
            self._save_settings()

    def _restore_settings(self) -> None:
        settings = load_settings()
        self.references = list(settings.reference_paths)
        if self.references:
            self.reference_label.config(
                text=f"{len(self.references)} restored: " + ", ".join(path.name for path in self.references[:3])
            )
            self._show_reference_thumbnails()
        if settings.search_folder:
            self.folder_var.set(str(settings.search_folder))
        if self.references and settings.search_folder:
            self.status_var.set("Previous reference photos and search folder restored.")
        elif self.references:
            self.status_var.set("Previous reference photos restored.")
        elif settings.search_folder:
            self.status_var.set("Last search folder restored.")

    def _save_settings(self) -> None:
        folder = Path(self.folder_var.get()) if self.folder_var.get() else None
        if folder is not None and not folder.is_dir():
            folder = None
        try:
            save_settings(AppSettings(tuple(self.references), folder))
        except OSError:
            # Remembering paths is a convenience and should never block a scan.
            pass

    def _close(self) -> None:
        self.cancel_event.set()
        self._save_settings()
        self.destroy()

    def _show_reference_thumbnails(self) -> None:
        for child in self.reference_thumbnails.winfo_children():
            child.destroy()
        self.reference_photos.clear()
        for index, path in enumerate(self.references[:6]):
            photo = self._thumbnail(path, (110, 82))
            self.reference_photos.append(photo)
            card = ttk.Frame(self.reference_thumbnails, padding=(0, 0, 8, 0))
            card.grid(row=0, column=index)
            ttk.Label(card, image=photo).pack()
            ttk.Label(card, text=path.name, width=16, anchor="center").pack()
        if len(self.references) > 6:
            ttk.Label(self.reference_thumbnails, text=f"+{len(self.references) - 6} more").grid(row=0, column=6, padx=8)

    def _thumbnail(self, path: Path, size: tuple[int, int]) -> ImageTk.PhotoImage:
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail(size, Image.Resampling.LANCZOS)
                background = Image.new("RGB", size, "#e6e6e6")
                offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
                background.paste(image, offset)
        except Exception:
            background = Image.new("RGB", size, "#e6e6e6")
            draw = ImageDraw.Draw(background)
            draw.line((10, 10, size[0] - 10, size[1] - 10), fill="#999999", width=3)
            draw.line((size[0] - 10, 10, 10, size[1] - 10), fill="#999999", width=3)
        return ImageTk.PhotoImage(background)

    def choose_folder(self) -> None:
        name = filedialog.askdirectory(title="Choose folder to scan")
        if name:
            self.folder_var.set(name)
            self._save_settings()

    def start_scan(self) -> None:
        folder = Path(self.folder_var.get())
        if not self.references:
            messagebox.showerror("Reference required", "Choose at least one reference photo.")
            return
        if not folder.is_dir():
            messagebox.showerror("Folder required", "Choose an existing folder to scan.")
            return
        self._save_settings()
        self.matches.clear()
        self.result_photos.clear()
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
            references = build_reference_embeddings(engine, self.references, self._request_face_selection)
            results = scan_folder(engine, folder, references, threshold, self._on_progress, self.cancel_event)
            self.events.put(("done", results))
        except Exception as exc:
            if self.cancel_event.is_set():
                self.events.put(("done", []))
            else:
                self.events.put(("error", str(exc)))

    def _on_progress(self, value: ScanProgress) -> None:
        self.events.put(("progress", value))

    def _request_face_selection(self, path: Path, faces: list[DetectedFace]) -> list[int]:
        request = FaceSelectionRequest(path, faces)
        self.events.put(("select_faces", request))
        request.ready.wait()
        return request.selected

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
                elif kind == "select_faces":
                    assert isinstance(payload, FaceSelectionRequest)
                    self._show_face_picker(payload)
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
        item_id = str(match.path)
        photo = self._thumbnail(match.path, (112, 78))
        self.result_photos[item_id] = photo
        self.tree.insert(
            "",
            "end",
            iid=item_id,
            image=photo,
            text=match.path.name,
            values=(f"{match.score:.3f}", match.face_count, str(match.path)),
        )

    def _finish(self, status: str) -> None:
        self.status_var.set(status)
        self.scan_button.config(state="normal")
        self.cancel_button.config(state="disabled")

    def _show_face_picker(self, request: FaceSelectionRequest) -> None:
        dialog = tk.Toplevel(self)
        self.face_dialog = dialog
        self.face_request = request
        dialog.title("Choose reference faces")
        dialog.transient(self)
        dialog.grab_set()
        dialog.protocol("WM_DELETE_WINDOW", lambda: finish([]))

        ttk.Label(
            dialog,
            text=f"{request.path.name} contains {len(request.faces)} faces. Select every person to search for.",
            padding=(14, 14, 14, 6),
        ).pack(anchor="w")
        grid = ttk.Frame(dialog, padding=(14, 6))
        grid.pack(fill="both", expand=True)
        variables: list[tk.BooleanVar] = []
        photos: list[ImageTk.PhotoImage] = []
        for index, face in enumerate(request.faces):
            rgb = cv2.cvtColor(face.preview, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((150, 150), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            photos.append(photo)
            variable = tk.BooleanVar(value=False)
            variables.append(variable)
            card = ttk.Frame(grid, padding=6, relief="ridge")
            card.grid(row=index // 4, column=index % 4, padx=5, pady=5, sticky="nsew")
            label = ttk.Label(card, image=photo, cursor="hand2")
            label.pack()
            check = ttk.Checkbutton(card, text=f"Face {index + 1}", variable=variable)
            check.pack(pady=(4, 0))
            label.bind("<Button-1>", lambda _event, value=variable: value.set(not value.get()))

        def finish(selected: list[int] | None = None) -> None:
            request.selected = selected if selected is not None else [i for i, value in enumerate(variables) if value.get()]
            request.ready.set()
            self.face_dialog = None
            self.face_request = None
            dialog.grab_release()
            dialog.destroy()

        buttons = ttk.Frame(dialog, padding=14)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Cancel selection", command=lambda: finish([])).pack(side="right")
        ttk.Button(buttons, text="Use selected faces", command=finish).pack(side="right", padx=8)
        dialog._face_photos = photos  # type: ignore[attr-defined]
        dialog.update_idletasks()
        dialog.geometry(f"+{self.winfo_rootx() + 50}+{self.winfo_rooty() + 50}")

    def cancel_scan(self) -> None:
        self.cancel_event.set()
        if self.face_request and self.face_dialog:
            self.face_request.selected = []
            self.face_request.ready.set()
            self.face_dialog.destroy()
            self.face_request = None
            self.face_dialog = None
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
