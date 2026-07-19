from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DETECTION_MAX_EDGE = 1280
MIN_PROFILE_SHARPNESS = 75.0


@dataclass(frozen=True)
class MatchResult:
    path: Path
    score: float
    face_count: int
    identified_count: int = 0
    cached: bool = False


@dataclass(frozen=True)
class ScanProgress:
    completed: int
    total: int
    path: Path
    match: MatchResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class DetectedFace:
    embedding: np.ndarray
    preview: np.ndarray
    bbox: tuple[int, int, int, int] | None = None
    sharpness: float = 0.0
    profile_eligible: bool = False


class FaceEngine:
    """Thin wrapper around OpenCV YuNet detection and SFace embeddings."""

    def __init__(self, detector_model: Path, recognizer_model: Path) -> None:
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_model), "", (320, 320), 0.75, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_model), "")

    def detect_faces(self, image_path: Path) -> list[DetectedFace]:
        image = read_image(image_path)
        detection_image, scale = resize_for_detection(image)
        height, width = detection_image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(detection_image)
        if faces is None:
            return []

        results: list[DetectedFace] = []
        for face in faces:
            face = restore_face_coordinates(face, scale)
            aligned = self.recognizer.alignCrop(image, face)
            feature = self.recognizer.feature(aligned).flatten().astype(np.float32)
            norm = float(np.linalg.norm(feature))
            if norm:
                # The aligned crop gives the picker a consistent, close-up preview.
                x, y, face_width, face_height = (int(round(value)) for value in face[:4])
                sharpness = face_sharpness(aligned)
                results.append(
                    DetectedFace(
                        feature / norm,
                        aligned,
                        (x, y, face_width, face_height),
                        sharpness,
                        sharpness >= MIN_PROFILE_SHARPNESS,
                    )
                )
        return results

    def embeddings(self, image_path: Path) -> list[np.ndarray]:
        return [face.embedding for face in self.detect_faces(image_path)]


def resize_for_detection(image: np.ndarray, max_edge: int = DETECTION_MAX_EDGE) -> tuple[np.ndarray, float]:
    """Shrink large photos so close-up faces stay within YuNet's detection range."""
    height, width = image.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale == 1.0:
        return image, scale
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def restore_face_coordinates(face: np.ndarray, scale: float) -> np.ndarray:
    """Map a YuNet box and its five landmarks back to the source resolution."""
    restored = face.copy()
    if scale != 1.0:
        restored[:14] /= scale
    return restored


def face_sharpness(aligned_face: np.ndarray) -> float:
    """Estimate focus using Laplacian variance on the normalized aligned crop."""
    gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


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
        if "screenshot" in path.name.casefold():
            continue
        resolved = path.resolve()
        if any(resolved == root or root in resolved.parents for root in excluded):
            continue
        files.append(path)
    return sorted(files, key=lambda item: str(item).casefold())


def build_reference_embeddings(
    engine: FaceEngine,
    reference_paths: Iterable[Path],
    select_faces: Callable[[Path, list[DetectedFace]], Iterable[int]] | None = None,
) -> list[np.ndarray]:
    embeddings: list[np.ndarray] = []
    failures: list[str] = []
    for path in reference_paths:
        try:
            faces = engine.detect_faces(path)
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")
            continue
        if len(faces) == 1:
            embeddings.append(faces[0].embedding)
        elif not faces:
            failures.append(f"{path.name}: no face found")
        elif select_faces:
            selected = list(select_faces(path, faces))
            valid = [index for index in selected if 0 <= index < len(faces)]
            embeddings.extend(faces[index].embedding for index in valid)
            if not valid:
                failures.append(f"{path.name}: no faces selected")
        else:
            failures.append(f"{path.name}: contains {len(faces)} faces; no face selector was provided")
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
