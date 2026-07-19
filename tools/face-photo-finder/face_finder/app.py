from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk

from .catalog import (
    FaceCatalog,
    KnownIdentity,
    UnknownGroup,
    best_known_identity,
    best_unknown_group,
    bounded_profile,
    default_catalog_path,
)
from .models import ensure_models
from .prefetch import DetectionPrefetcher
from .scanner import (
    DetectedFace,
    FaceEngine,
    MatchResult,
    ScanProgress,
    best_similarity,
    build_reference_embeddings,
    image_files,
    read_image,
)
from .settings import AppSettings, load_settings, save_settings


class FaceSelectionRequest:
    def __init__(self, path: Path, faces: list[DetectedFace]) -> None:
        self.path = path
        self.faces = faces
        self.selected: list[int] = []
        self.ready = threading.Event()


@dataclass(frozen=True)
class ContextFace:
    bbox: tuple[int, int, int, int]
    status: str
    label: str


class IdentityRequest:
    def __init__(
        self,
        path: Path,
        preview: np.ndarray,
        names: list[str],
        bbox: tuple[int, int, int, int] | None = None,
        sharpness: float | None = None,
        profile_eligible: bool = True,
        context_faces: list[ContextFace] | None = None,
    ) -> None:
        self.path = path
        self.preview = preview
        self.names = names
        self.bbox = bbox
        self.sharpness = sharpness
        self.profile_eligible = profile_eligible
        self.context_faces = context_faces or (
            [ContextFace(bbox, "current", "Person to identify")] if bbox else []
        )
        self.name: str | None = None
        self.skip_remaining = False
        self.not_a_face = False
        self.intentionally_unknown = False
        self.save_as_art = False
        self.target_embedding: np.ndarray | None = None
        self.ready = threading.Event()
        self.related_faces: list[tuple[Path, np.ndarray]] = []
        self.related_lock = threading.Lock()

    def add_related_face(self, path: Path, preview: np.ndarray) -> bool:
        with self.related_lock:
            if len(self.related_faces) >= 12 or any(existing_path == path for existing_path, _ in self.related_faces):
                return False
            self.related_faces.append((path, preview))
            return True


def add_known_sample(
    identities: list[KnownIdentity], identity_id: int, name: str, embedding: np.ndarray
) -> list[KnownIdentity]:
    updated: list[KnownIdentity] = []
    found = False
    for identity in identities:
        if identity.identity_id == identity_id:
            updated.append(KnownIdentity(identity_id, identity.name, identity.embeddings + (embedding,)))
            found = True
        else:
            updated.append(identity)
    if not found:
        updated.append(KnownIdentity(identity_id, name, (embedding,)))
    return [KnownIdentity(item.identity_id, item.name, bounded_profile(item.embeddings)) for item in updated]


def add_unknown_sample(
    groups: list[UnknownGroup], group_id: int, embedding: np.ndarray
) -> list[UnknownGroup]:
    updated: list[UnknownGroup] = []
    found = False
    for group in groups:
        if group.group_id == group_id:
            updated.append(UnknownGroup(group_id, bounded_profile(group.embeddings + (embedding,))))
            found = True
        else:
            updated.append(group)
    if not found:
        updated.append(UnknownGroup(group_id, (embedding,)))
    return updated


def detection_context(
    faces: list[DetectedFace],
    current_index: int,
    states: dict[int, tuple[str, str]],
) -> list[ContextFace]:
    annotations: list[ContextFace] = []
    for index, face in enumerate(faces):
        if face.bbox is None or states.get(index, ("", ""))[0] == "not_face":
            continue
        if index == current_index:
            annotations.append(ContextFace(face.bbox, "current", "Person to identify"))
        else:
            status, label = states.get(index, ("unprocessed", "Unprocessed"))
            annotations.append(ContextFace(face.bbox, status, label))
    return annotations


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
        self.gallery_photos: dict[str, ImageTk.PhotoImage] = {}
        self.gallery_selected: dict[str, tk.BooleanVar] = {}
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.face_dialog: tk.Toplevel | None = None
        self.face_request: FaceSelectionRequest | None = None
        self.identity_dialog: tk.Toplevel | None = None
        self.identity_request: IdentityRequest | None = None
        self.prefetcher: DetectionPrefetcher | None = None
        self.folder_var = tk.StringVar()
        self.known_person_var = tk.StringVar()
        self.results_view_var = tk.StringVar(value="details")
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
        reference_actions = ttk.Frame(outer)
        reference_actions.grid(row=1, column=1, sticky="n")
        ttk.Button(reference_actions, text="Choose photos…", command=self.choose_references).pack(fill="x")
        ttk.Button(reference_actions, text="Clear photos", command=self.clear_references).pack(fill="x", pady=(4, 10))
        ttk.Label(reference_actions, text="Or choose a known person").pack(anchor="w")
        self.known_person_combo = ttk.Combobox(
            reference_actions, textvariable=self.known_person_var, state="readonly", width=24
        )
        self.known_person_combo.pack(fill="x", pady=(3, 0))
        self._refresh_known_people()

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

        view_bar = ttk.Frame(outer)
        view_bar.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(view_bar, text="Results view:").pack(side="left")
        ttk.Radiobutton(
            view_bar, text="Details", value="details", variable=self.results_view_var, command=self._toggle_results_view
        ).pack(side="left", padx=(8, 2))
        ttk.Radiobutton(
            view_bar, text="Gallery", value="gallery", variable=self.results_view_var, command=self._toggle_results_view
        ).pack(side="left", padx=2)

        self.results_container = ttk.Frame(outer)
        self.results_container.grid(row=8, column=0, columnspan=2, sticky="nsew")
        self.results_container.columnconfigure(0, weight=1)
        self.results_container.rowconfigure(0, weight=1)

        style = ttk.Style(self)
        style.configure("Results.Treeview", rowheight=84)
        self.tree = ttk.Treeview(
            self.results_container,
            columns=("score", "faces", "identified", "cached", "path"),
            show="tree headings",
            selectmode="extended",
            style="Results.Treeview",
        )
        self.tree.heading("#0", text="Preview")
        self.tree.heading("score", text="Similarity")
        self.tree.heading("faces", text="Faces")
        self.tree.heading("identified", text="Identified")
        self.tree.heading("cached", text="Cached")
        self.tree.heading("path", text="Photo")
        self.tree.column("#0", width=120, minwidth=120, anchor="center", stretch=False)
        self.tree.column("score", width=90, anchor="center", stretch=False)
        self.tree.column("faces", width=60, anchor="center", stretch=False)
        self.tree.column("identified", width=75, anchor="center", stretch=False)
        self.tree.column("cached", width=65, anchor="center", stretch=False)
        self.tree.column("path", width=700)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<Double-1>", lambda _event: self.open_selected())

        self.gallery_canvas = tk.Canvas(self.results_container, highlightthickness=0)
        self.gallery_scrollbar = ttk.Scrollbar(
            self.results_container, orient="vertical", command=self.gallery_canvas.yview
        )
        self.gallery_canvas.configure(yscrollcommand=self.gallery_scrollbar.set)
        self.gallery_frame = ttk.Frame(self.gallery_canvas)
        self.gallery_window = self.gallery_canvas.create_window((0, 0), window=self.gallery_frame, anchor="nw")
        self.gallery_frame.bind(
            "<Configure>", lambda _event: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
        )
        self.gallery_canvas.bind(
            "<Configure>", lambda event: self.gallery_canvas.itemconfigure(self.gallery_window, width=event.width)
        )
        self.gallery_canvas.bind("<MouseWheel>", lambda event: self.gallery_canvas.yview_scroll(-event.delta // 120, "units"))

        actions = ttk.Frame(outer)
        actions.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(actions, text="Open selected", command=self.open_selected).pack(side="left")
        ttk.Button(actions, text="Export CSV…", command=self.export_csv).pack(side="left", padx=8)
        ttk.Button(actions, text="Copy matches…", command=self.copy_matches).pack(side="left")
        self.reset_button = ttk.Button(actions, text="Reset face database…", command=self.reset_database)
        self.reset_button.pack(side="left", padx=8)
        ttk.Label(actions, text="Verify matches before relying on them; face recognition can be wrong.").pack(side="right")

        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(8, weight=1)

    def choose_references(self) -> None:
        names = filedialog.askopenfilenames(title="Choose reference photos", filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff")])
        if names:
            self.references = [Path(name) for name in names]
            self.reference_label.config(text=f"{len(self.references)} selected: " + ", ".join(path.name for path in self.references[:3]))
            self._show_reference_thumbnails()
            self._save_settings()

    def clear_references(self) -> None:
        self.references.clear()
        self.reference_label.config(text="None selected — catalog all faces or choose a known person")
        self._show_reference_thumbnails()
        self._save_settings()

    def _refresh_known_people(self, names: list[str] | None = None) -> None:
        if names is None:
            catalog = FaceCatalog(default_catalog_path())
            try:
                names = [identity.name for identity in catalog.identities()]
            finally:
                catalog.close()
        current = self.known_person_var.get()
        values = [""] + names
        self.known_person_combo["values"] = values
        if current not in values:
            self.known_person_var.set("")

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
        if not folder.is_dir():
            messagebox.showerror("Folder required", "Choose an existing folder to scan.")
            return
        self._save_settings()
        self.matches.clear()
        self.result_photos.clear()
        self.gallery_photos.clear()
        self.gallery_selected.clear()
        self.tree.delete(*self.tree.get_children())
        for child in self.gallery_frame.winfo_children():
            child.destroy()
        self.cancel_event.clear()
        self.scan_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.reset_button.config(state="disabled")
        self.progress["value"] = 0
        threshold = self.threshold_var.get()
        references = list(self.references)
        known_person = self.known_person_var.get().strip()
        self.worker = threading.Thread(
            target=self._scan_worker, args=(folder, threshold, references, known_person), daemon=True
        )
        self.worker.start()

    def _scan_worker(
        self, folder: Path, threshold: float, reference_paths: list[Path], known_person: str
    ) -> None:
        catalog: FaceCatalog | None = None
        try:
            model_dir = Path(__file__).resolve().parents[1] / "models"
            detector, recognizer = ensure_models(model_dir, lambda value: self.events.put(("status", value)))
            engine = FaceEngine(detector, recognizer)
            catalog = FaceCatalog(default_catalog_path())
            known_identities = catalog.identities()
            unknown_groups = catalog.unknown_groups()
            selected_identity = next((item for item in known_identities if item.name == known_person), None)
            catalog_all = not reference_paths and selected_identity is None
            target_identity_id = selected_identity.identity_id if selected_identity else None
            if selected_identity:
                references = list(selected_identity.embeddings)
                if not references:
                    raise ValueError(f"{known_person} has no learned face samples yet.")
            elif reference_paths:
                self.events.put(("status", "Reading reference faces…"))
                references = build_reference_embeddings(engine, reference_paths, self._request_face_selection)
            else:
                references = []
                self.events.put(("status", "Cataloging every photo that contains a face…"))

            files = image_files(folder)
            uncached_paths = [path for path in files if catalog.cached_image(path) is None]
            self.prefetcher = DetectionPrefetcher(
                uncached_paths,
                lambda: FaceEngine(detector, recognizer),
                self.cancel_event,
                self._on_prefetched_faces,
            )
            self.prefetcher.start()
            results: list[MatchResult] = []
            skip_unknowns = False
            for index, path in enumerate(files, start=1):
                if self.cancel_event.is_set():
                    break
                error: str | None = None
                match: MatchResult | None = None
                try:
                    cached = catalog.cached_image(path)
                    if cached:
                        catalog_faces = catalog.faces_for_image(cached.image_id)
                        if catalog_all and not skip_unknowns:
                            for face in catalog_faces:
                                if face.identity_id is not None or face.intentionally_unknown or face.preview is None:
                                    continue
                                preview = cv2.imdecode(face.preview, cv2.IMREAD_COLOR)
                                if preview is None:
                                    continue
                                context_faces = []
                                for context_face in catalog_faces:
                                    if context_face.bbox is None:
                                        continue
                                    if context_face.face_id == face.face_id:
                                        context_faces.append(
                                            ContextFace(context_face.bbox, "current", "Person to identify")
                                        )
                                    elif context_face.identity_id is not None:
                                        context_faces.append(
                                            ContextFace(
                                                context_face.bbox,
                                                "identified",
                                                context_face.identity_name or "Identified",
                                            )
                                        )
                                    elif context_face.intentionally_unknown:
                                        context_faces.append(
                                            ContextFace(context_face.bbox, "unknown", "Unknown person")
                                        )
                                    else:
                                        context_faces.append(
                                            ContextFace(context_face.bbox, "unprocessed", "Unprocessed")
                                        )
                                name, stop_asking, not_a_face, intentionally_unknown, save_as_art = self._request_identity(
                                    path, preview, known_identities, face.bbox, face.sharpness,
                                    face.profile_eligible, context_faces, face.embedding,
                                )
                                skip_unknowns = skip_unknowns or stop_asking
                                if not_a_face:
                                    catalog.remove_face(face.face_id)
                                elif intentionally_unknown:
                                    group_id = catalog.mark_intentionally_unknown(face.face_id)
                                    if face.profile_eligible:
                                        unknown_groups = add_unknown_sample(
                                            unknown_groups, group_id, face.embedding
                                        )
                                elif name:
                                    identity_id = catalog.get_or_create_identity(name)
                                    catalog.assign_face(face.face_id, identity_id, as_art=save_as_art)
                                    known_identities = catalog.identities()
                                if skip_unknowns:
                                    break
                        catalog_faces = catalog.faces_for_image(cached.image_id)
                        candidate_embeddings = [face.embedding for face in catalog_faces]
                        score = best_similarity(references, candidate_embeddings) if references else -1.0
                        if target_identity_id is not None:
                            for face in catalog_faces:
                                if face.identity_id is None and best_similarity(references, [face.embedding]) >= threshold:
                                    catalog.assign_face(face.face_id, target_identity_id)
                        if catalog_faces and (catalog_all or score >= threshold):
                            refreshed = catalog.cached_image(path) or cached
                            match = MatchResult(
                                path, score, refreshed.face_count, refreshed.identified_count, cached=True
                            )
                    else:
                        detected = self.prefetcher.get(path)
                        accepted_faces: list[DetectedFace] = []
                        assignments: list[int | None] = []
                        unknown_statuses: list[bool] = []
                        unknown_group_ids: list[int | None] = []
                        profile_eligible_flags: list[bool] = []
                        art_flags: list[bool] = []
                        review_states: dict[int, tuple[str, str]] = {}
                        for detected_index, face in enumerate(detected):
                            learning_threshold = max(threshold, 0.55)
                            identity_id, _identity_score = best_known_identity(
                                face.embedding, known_identities, learning_threshold
                            )
                            if identity_id is not None and face.profile_eligible:
                                identity = next(
                                    item for item in known_identities if item.identity_id == identity_id
                                )
                                known_identities = add_known_sample(
                                    known_identities, identity_id, identity.name, face.embedding
                                )
                            if identity_id is not None:
                                identity_name = next(
                                    item.name for item in known_identities if item.identity_id == identity_id
                                )
                                review_states[detected_index] = ("identified", identity_name)
                            unknown_group_id: int | None = None
                            intentionally_unknown = False
                            save_as_art = False
                            if identity_id is None:
                                unknown_group_id, _unknown_score = best_unknown_group(
                                    face.embedding, unknown_groups, learning_threshold
                                )
                                intentionally_unknown = unknown_group_id is not None
                                if unknown_group_id is not None and face.profile_eligible:
                                    unknown_groups = add_unknown_sample(
                                        unknown_groups, unknown_group_id, face.embedding
                                    )
                                if unknown_group_id is not None:
                                    review_states[detected_index] = ("unknown", "Unknown person")
                            if (
                                identity_id is None
                                and unknown_group_id is None
                                and catalog_all
                                and not skip_unknowns
                            ):
                                name, stop_asking, not_a_face, intentionally_unknown, save_as_art = self._request_identity(
                                    path, face.preview, known_identities, face.bbox, face.sharpness,
                                    face.profile_eligible,
                                    detection_context(detected, detected_index, review_states),
                                    face.embedding,
                                )
                                skip_unknowns = skip_unknowns or stop_asking
                                if not_a_face:
                                    review_states[detected_index] = ("not_face", "Not a face")
                                    continue
                                if name:
                                    identity_id = catalog.get_or_create_identity(name)
                                    if face.profile_eligible and not save_as_art:
                                        known_identities = add_known_sample(
                                            known_identities, identity_id, name, face.embedding
                                        )
                                    review_states[detected_index] = ("identified", name)
                                elif intentionally_unknown:
                                    unknown_group_id = catalog.create_unknown_group()
                                    if face.profile_eligible:
                                        unknown_groups = add_unknown_sample(
                                            unknown_groups, unknown_group_id, face.embedding
                                        )
                                    review_states[detected_index] = ("unknown", "Unknown person")
                                else:
                                    review_states[detected_index] = ("unprocessed", "Unprocessed")
                            accepted_faces.append(face)
                            assignments.append(identity_id)
                            unknown_statuses.append(intentionally_unknown and identity_id is None)
                            unknown_group_ids.append(unknown_group_id if identity_id is None else None)
                            profile_eligible_flags.append(face.profile_eligible and not save_as_art)
                            art_flags.append(save_as_art)
                        detected = accepted_faces
                        previews = [cv2.imencode(".jpg", face.preview)[1] for face in detected]
                        stored = catalog.store_scan(
                            path, [face.embedding for face in detected], assignments, previews, unknown_statuses,
                            [face.bbox for face in detected], unknown_group_ids,
                            [face.sharpness for face in detected],
                            profile_eligible_flags,
                            art_flags,
                        )
                        candidate_embeddings = [face.embedding for face in detected]
                        score = best_similarity(references, candidate_embeddings) if references else -1.0
                        if target_identity_id is not None and score >= threshold:
                            for face_index, face in enumerate(detected):
                                if assignments[face_index] is None and best_similarity(references, [face.embedding]) >= threshold:
                                    assignments[face_index] = target_identity_id
                                    unknown_statuses[face_index] = False
                                    unknown_group_ids[face_index] = None
                            if assignments != [face.identity_id for face in catalog.faces_for_image(stored.image_id)]:
                                catalog.store_scan(
                                    path, [face.embedding for face in detected], assignments, previews,
                                    unknown_statuses, [face.bbox for face in detected], unknown_group_ids,
                                    [face.sharpness for face in detected],
                                    profile_eligible_flags,
                                    art_flags,
                                )
                        if detected and (catalog_all or score >= threshold):
                            refreshed = catalog.cached_image(path) or stored
                            match = MatchResult(
                                path, score, len(detected), refreshed.identified_count, cached=False
                            )
                    if match:
                        results.append(match)
                except Exception as exc:
                    error = str(exc)
                self._on_progress(ScanProgress(index, len(files), path, match, error))
            results.sort(key=lambda item: (-item.score, str(item.path).casefold()))
            self.events.put(("identity_names", [item.name for item in catalog.identities()]))
            self.events.put(("done", results))
        except Exception as exc:
            if self.cancel_event.is_set():
                self.events.put(("done", []))
            else:
                self.events.put(("error", str(exc)))
        finally:
            if self.prefetcher:
                self.prefetcher.stop()
                self.prefetcher = None
            if catalog:
                catalog.close()

    def _on_progress(self, value: ScanProgress) -> None:
        self.events.put(("progress", value))

    def _request_face_selection(self, path: Path, faces: list[DetectedFace]) -> list[int]:
        request = FaceSelectionRequest(path, faces)
        self.events.put(("select_faces", request))
        request.ready.wait()
        return request.selected

    def _request_identity(
        self,
        path: Path,
        preview: np.ndarray,
        identities: list[object],
        bbox: tuple[int, int, int, int] | None,
        sharpness: float | None,
        profile_eligible: bool,
        context_faces: list[ContextFace],
        target_embedding: np.ndarray,
    ) -> tuple[str | None, bool, bool, bool, bool]:
        names = [str(getattr(identity, "name")) for identity in identities]
        request = IdentityRequest(
            path, preview, names, bbox, sharpness, profile_eligible, context_faces
        )
        request.target_embedding = target_embedding
        request.add_related_face(path, preview)
        self.identity_request = request
        if self.prefetcher:
            for related_path, face in self.prefetcher.buffered_faces():
                if float(np.dot(target_embedding, face.embedding)) >= 0.55:
                    request.add_related_face(related_path, face.preview)
        self.events.put(("identify_face", request))
        request.ready.wait()
        if self.identity_request is request:
            self.identity_request = None
        return (
            request.name,
            request.skip_remaining,
            request.not_a_face,
            request.intentionally_unknown,
            request.save_as_art,
        )

    def _on_prefetched_faces(self, path: Path, faces: list[DetectedFace]) -> None:
        request = self.identity_request
        if request is None:
            return
        target = getattr(request, "target_embedding", None)
        if target is None:
            return
        for face in faces:
            if float(np.dot(target, face.embedding)) >= 0.55 and request.add_related_face(path, face.preview):
                self.events.put(("related_face", request))

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
                elif kind == "identify_face":
                    assert isinstance(payload, IdentityRequest)
                    self._show_identity_prompt(payload)
                elif kind == "related_face":
                    if payload is self.identity_request and self.identity_dialog:
                        self._refresh_related_faces(payload)  # type: ignore[arg-type]
                elif kind == "identity_names":
                    self._refresh_known_people(list(payload))  # type: ignore[arg-type]
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
            values=(
                f"{match.score:.3f}" if match.score >= 0 else "All faces",
                match.face_count,
                match.identified_count,
                "Yes" if match.cached else "No",
                str(match.path),
            ),
        )
        self._add_gallery_match(match)

    def _add_gallery_match(self, match: MatchResult) -> None:
        item_id = str(match.path)
        index = len(self.gallery_selected)
        photo = self._thumbnail(match.path, (210, 150))
        selected = tk.BooleanVar(value=False)
        self.gallery_photos[item_id] = photo
        self.gallery_selected[item_id] = selected

        card = ttk.Frame(self.gallery_frame, padding=8, relief="ridge")
        card.grid(row=index // 3, column=index % 3, padx=6, pady=6, sticky="nsew")
        self.gallery_frame.columnconfigure(index % 3, weight=1)
        preview = ttk.Label(card, image=photo, cursor="hand2")
        preview.pack()
        ttk.Checkbutton(card, text=match.path.name, variable=selected).pack(anchor="w", pady=(5, 0))
        score_text = f"Similarity {match.score:.3f}" if match.score >= 0 else "All faces"
        ttk.Label(
            card,
            text=f"{score_text} • {match.identified_count}/{match.face_count} identified"
            + (" • cached" if match.cached else ""),
        ).pack(anchor="w")
        preview.bind("<Button-1>", lambda _event, value=selected: value.set(not value.get()))
        preview.bind("<Double-1>", lambda _event, path=match.path: self._open_path(path))

    def _toggle_results_view(self) -> None:
        if self.results_view_var.get() == "gallery":
            self.tree.grid_remove()
            self.gallery_canvas.grid(row=0, column=0, sticky="nsew")
            self.gallery_scrollbar.grid(row=0, column=1, sticky="ns")
        else:
            self.gallery_canvas.grid_remove()
            self.gallery_scrollbar.grid_remove()
            self.tree.grid()

    def _finish(self, status: str) -> None:
        self.status_var.set(status)
        self.scan_button.config(state="normal")
        self.cancel_button.config(state="disabled")
        self.reset_button.config(state="normal")

    def reset_database(self) -> None:
        confirmed = messagebox.askyesno(
            "DANGER — Reset face database?",
            "This permanently deletes all learned names, anonymous people, face embeddings, cached scans, "
            "face previews, and scan statistics.\n\nYour original photos and saved folder/reference paths are not deleted. "
            "This action cannot be undone.\n\nReset the database now?",
            icon="warning",
            parent=self,
        )
        if not confirmed:
            return
        typed_confirmation = simpledialog.askstring(
            "DANGER — Permanent deletion",
            "This is the final confirmation.\n\nType RESET exactly to permanently erase the face database:",
            parent=self,
        )
        if typed_confirmation != "RESET":
            self.status_var.set("Database reset cancelled; confirmation text did not match RESET.")
            return
        try:
            catalog = FaceCatalog(default_catalog_path())
            try:
                catalog.reset()
            finally:
                catalog.close()
        except Exception as exc:
            messagebox.showerror("Reset failed", str(exc), parent=self)
            return
        self.known_person_var.set("")
        self._refresh_known_people([])
        self.matches.clear()
        self.result_photos.clear()
        self.gallery_photos.clear()
        self.gallery_selected.clear()
        self.tree.delete(*self.tree.get_children())
        for child in self.gallery_frame.winfo_children():
            child.destroy()
        self.progress["value"] = 0
        self.status_var.set("Face database reset. Source photos and preferences were not changed.")
        messagebox.showinfo(
            "Database reset",
            "All biometric identities and cached scan data have been removed. Your source photos were not changed.",
            parent=self,
        )

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

    def _show_identity_prompt(self, request: IdentityRequest) -> None:
        dialog = tk.Toplevel(self)
        self.identity_dialog = dialog
        self.identity_request = request
        dialog.title("Identify unfamiliar face")
        dialog.transient(self)
        dialog.grab_set()

        content = ttk.Frame(dialog, padding=16)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text=f"Who is this person in {request.path.name}?").pack(anchor="w", pady=(0, 10))
        if not request.profile_eligible:
            ttk.Label(
                content,
                text="This face is blurry. It can be linked to an identity, but it will not train the biometric profile.",
                foreground="#a05a00",
                wraplength=440,
            ).pack(anchor="w", pady=(0, 10))
        rgb = cv2.cvtColor(request.preview, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((240, 240), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        ttk.Label(content, image=photo).pack()
        ttk.Label(content, text="Possible appearances detected so far:").pack(anchor="w", pady=(12, 3))
        related_frame = ttk.Frame(content)
        related_frame.pack(fill="x")
        ttk.Label(content, text="Choose an existing person or type a new name:").pack(anchor="w", pady=(12, 3))
        name_var = tk.StringVar()
        name_box = ttk.Combobox(content, textvariable=name_var, values=request.names, width=38)
        name_box.pack(fill="x")
        name_box.focus_set()

        def finish(
            name: str | None,
            skip_remaining: bool = False,
            not_a_face: bool = False,
            intentionally_unknown: bool = False,
            save_as_art: bool = False,
        ) -> None:
            request.name = name
            request.skip_remaining = skip_remaining
            request.not_a_face = not_a_face
            request.intentionally_unknown = intentionally_unknown
            request.save_as_art = save_as_art
            request.ready.set()
            self.identity_dialog = None
            self.identity_request = None
            dialog.grab_release()
            dialog.destroy()

        def save(save_as_art: bool = False) -> None:
            name = " ".join(name_var.get().split())
            if not name:
                messagebox.showwarning("Name required", "Enter a name or use one of the skip options.", parent=dialog)
                return
            finish(name, save_as_art=save_as_art)

        buttons = ttk.Frame(content)
        buttons.pack(fill="x", pady=(14, 0))
        ttk.Button(buttons, text="Save identity", command=save).pack(side="right")
        ttk.Button(buttons, text="Save identity as art", command=lambda: save(True)).pack(
            side="right", padx=8
        )
        ttk.Button(buttons, text="Skip this face", command=lambda: finish(None)).pack(side="right", padx=8)
        ttk.Button(buttons, text="Not a face", command=lambda: finish(None, not_a_face=True)).pack(side="right")
        ttk.Button(
            buttons,
            text="I don't know this person",
            command=lambda: finish(None, intentionally_unknown=True),
        ).pack(side="right", padx=8)
        ttk.Button(buttons, text="Skip remaining", command=lambda: finish(None, True)).pack(side="left")
        ttk.Button(buttons, text="View context", command=lambda: self._show_context_image(request)).pack(
            side="left", padx=8
        )
        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(None))
        dialog._identity_photo = photo  # type: ignore[attr-defined]
        dialog._related_frame = related_frame  # type: ignore[attr-defined]
        dialog._related_photos = []  # type: ignore[attr-defined]
        self._refresh_related_faces(request)
        dialog.bind("<Return>", lambda _event: save())
        dialog.update_idletasks()
        dialog.geometry(f"+{self.winfo_rootx() + 80}+{self.winfo_rooty() + 80}")

    def _refresh_related_faces(self, request: IdentityRequest) -> None:
        dialog = self.identity_dialog
        if not dialog or not dialog.winfo_exists():
            return
        frame = dialog._related_frame  # type: ignore[attr-defined]
        for child in frame.winfo_children():
            child.destroy()
        photos: list[ImageTk.PhotoImage] = []
        with request.related_lock:
            related = list(request.related_faces)
        for index, (path, preview) in enumerate(related[:8]):
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((92, 92), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            photos.append(photo)
            card = ttk.Frame(frame, padding=(0, 0, 6, 0))
            card.grid(row=index // 4, column=index % 4, sticky="n")
            ttk.Label(card, image=photo).pack()
            ttk.Label(card, text=path.name, width=14, anchor="center").pack()
        dialog._related_photos = photos  # type: ignore[attr-defined]

    def _show_context_image(self, request: IdentityRequest) -> tk.Toplevel | None:
        try:
            source = read_image(request.path)
        except Exception as exc:
            messagebox.showerror("Unable to open image", str(exc), parent=self.identity_dialog or self)
            return None

        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_width = min(1200, max(600, self.winfo_screenwidth() - 160))
        max_height = min(820, max(450, self.winfo_screenheight() - 200))
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        if request.context_faces:
            source_height, source_width = source.shape[:2]
            scale_x = image.width / source_width
            scale_y = image.height / source_height
            draw = ImageDraw.Draw(image)
            colors = {
                "unprocessed": "#ff3030",
                "current": "#00ffff",
                "identified": "#20d060",
                "unknown": "#9a9a9a",
            }
            # Draw the active cyan target last so it remains visually dominant.
            annotations = sorted(request.context_faces, key=lambda item: item.status == "current")
            for annotation in annotations:
                x, y, width, height = annotation.bbox
                left = max(0, round(x * scale_x))
                top = max(0, round(y * scale_y))
                right = min(image.width - 1, round((x + width) * scale_x))
                bottom = min(image.height - 1, round((y + height) * scale_y))
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
                color = colors.get(annotation.status, colors["unprocessed"])
                for outline_color, line_width in (("#101010", 8), (color, 4)):
                    draw.rectangle((left, top, right, bottom), outline=outline_color, width=line_width)
                radius = max(9, min(24, max(1, right - left) // 8))
                draw.ellipse(
                    (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                    outline=color,
                    width=3,
                )
                draw.line(
                    (center_x - radius * 2, center_y, center_x + radius * 2, center_y), fill=color, width=3
                )
                draw.line(
                    (center_x, center_y - radius * 2, center_x, center_y + radius * 2), fill=color, width=3
                )
                label_y = max(0, top - 20)
                label_box = draw.textbbox((left, label_y), annotation.label)
                draw.rectangle(label_box, fill="#101010")
                draw.text((left, label_y), annotation.label, fill=color)

        viewer = tk.Toplevel(self)
        viewer.title(f"Context — {request.path.name}")
        viewer.transient(self.identity_dialog or self)
        frame = ttk.Frame(viewer, padding=10)
        frame.pack(fill="both", expand=True)
        photo = ImageTk.PhotoImage(image)
        ttk.Label(frame, image=photo).pack()
        ttk.Label(
            frame,
            text="Red: unprocessed   •   Cyan: identifying now   •   Green: identified   •   Gray: unknown",
        ).pack(anchor="w", pady=(8, 0))
        ttk.Label(frame, text=str(request.path), wraplength=max_width).pack(anchor="w", pady=(8, 4))
        def close_viewer() -> None:
            viewer.grab_release()
            viewer.destroy()
            if self.identity_dialog and self.identity_dialog.winfo_exists():
                self.identity_dialog.grab_set()

        ttk.Button(frame, text="Close", command=close_viewer).pack(anchor="e")
        viewer._context_photo = photo  # type: ignore[attr-defined]
        viewer.protocol("WM_DELETE_WINDOW", close_viewer)
        viewer.grab_set()
        viewer.update_idletasks()
        viewer.geometry(f"+{self.winfo_rootx() + 30}+{self.winfo_rooty() + 30}")
        return viewer

    def cancel_scan(self) -> None:
        self.cancel_event.set()
        if self.face_request and self.face_dialog:
            self.face_request.selected = []
            self.face_request.ready.set()
            self.face_dialog.destroy()
            self.face_request = None
            self.face_dialog = None
        if self.identity_request and self.identity_dialog:
            self.identity_request.skip_remaining = True
            self.identity_request.ready.set()
            self.identity_dialog.destroy()
            self.identity_request = None
            self.identity_dialog = None
        self.status_var.set("Cancelling after the current photo…")

    def selected_paths(self) -> list[Path]:
        if self.results_view_var.get() == "gallery":
            selected = [Path(path) for path, value in self.gallery_selected.items() if value.get()]
            return selected or [Path(path) for path in self.gallery_selected]
        selected = self.tree.selection() or self.tree.get_children()
        return [Path(self.tree.item(item, "values")[4]) for item in selected]

    def open_selected(self) -> None:
        paths = self.selected_paths()
        if paths:
            self._open_path(paths[0])

    def _open_path(self, path: Path) -> None:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def export_csv(self) -> None:
        if not self.matches:
            messagebox.showinfo("Nothing to export", "Run a scan that finds matches first.")
            return
        name = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if name:
            with Path(name).open("w", newline="", encoding="utf-8-sig") as output:
                writer = csv.writer(output)
                writer.writerow(["path", "similarity", "faces_detected", "faces_identified", "cached"])
                writer.writerows(
                    (str(item.path), f"{item.score:.6f}" if item.score >= 0 else "", item.face_count,
                     item.identified_count, item.cached)
                    for item in self.matches
                )
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
