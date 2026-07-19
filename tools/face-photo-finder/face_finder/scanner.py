from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MatchResult:
    path: Path
    score: float
    face_count: int


@dataclass(frozen=True)
class ScanProgress:
    completed: int
    total: int
    path: Path
    match: MatchResult | None = None
    error: str | None = None


class FaceEngine:
    """Thin wrapper around OpenCV YuNet detection and SFace embeddings."""

    def __init__(self, detector_model: Path, recognizer_model: Path) -> None:
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_model), "", (320, 320), 0.75, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_model), "")

    def embeddings(self, image_path: Path) -> list[np.ndarray]:
        image = read_image(image_path)
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None:
            return []

        results: list[np.ndarray] = []
        for face in faces:
            aligned = self.recognizer.alignCrop(image, face)
            feature = self.recognizer.feature(aligned).flatten().astype(np.float32)
            norm = float(np.linalg.norm(feature))
            if norm:
                results.append(feature / norm)
        return results


def read_image(path: Path) -> np.ndarray:
    """Read paths containing non-ASCII characters on Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("unsupported or damaged image")
    return image


def image_files(folder: Path, excluded_roots: Iterable[Path] = ()) -> list[Path]:
    excluded = [root.resolve() for root in excluded_roots]
    files: list[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved = path.resolve()
        if any(resolved == root or root in resolved.parents for root in excluded):
            continue
        files.append(path)
    return sorted(files, key=lambda item: str(item).casefold())


def build_reference_embeddings(engine: FaceEngine, reference_paths: Iterable[Path]) -> list[np.ndarray]:
    embeddings: list[np.ndarray] = []
    failures: list[str] = []
    for path in reference_paths:
        try:
            faces = engine.embeddings(path)
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")
            continue
        if len(faces) == 1:
            embeddings.append(faces[0])
        elif not faces:
            failures.append(f"{path.name}: no face found")
        else:
            failures.append(f"{path.name}: contains {len(faces)} faces; use a photo with one face")
    if not embeddings:
        detail = "; ".join(failures) or "no reference images selected"
        raise ValueError(f"No usable reference faces. {detail}")
    return embeddings


def best_similarity(references: Iterable[np.ndarray], candidates: Iterable[np.ndarray]) -> float:
    refs = list(references)
    faces = list(candidates)
    if not refs or not faces:
        return -1.0
    return max(float(np.dot(reference, candidate)) for reference in refs for candidate in faces)


def scan_folder(
    engine: FaceEngine,
    folder: Path,
    references: list[np.ndarray],
    threshold: float = 0.45,
    progress: Callable[[ScanProgress], None] | None = None,
    cancel: threading.Event | None = None,
    excluded_roots: Iterable[Path] = (),
) -> list[MatchResult]:
    files = image_files(folder, excluded_roots)
    matches: list[MatchResult] = []
    for index, path in enumerate(files, start=1):
        if cancel and cancel.is_set():
            break
        error = None
        match = None
        try:
            faces = engine.embeddings(path)
            score = best_similarity(references, faces)
            if score >= threshold:
                match = MatchResult(path, score, len(faces))
                matches.append(match)
        except Exception as exc:
            error = str(exc)
        if progress:
            progress(ScanProgress(index, len(files), path, match, error))
    return sorted(matches, key=lambda item: (-item.score, str(item.path).casefold()))
