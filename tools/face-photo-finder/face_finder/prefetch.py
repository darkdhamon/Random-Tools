from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import threading

from .scanner import DetectedFace, FaceEngine


class DetectionPrefetcher:
    """Detect faces ahead of the catalog worker using an independent face engine."""

    def __init__(
        self,
        paths: list[Path],
        engine_factory: Callable[[], FaceEngine],
        cancel: threading.Event,
        on_detected: Callable[[Path, list[DetectedFace]], None] | None = None,
        max_buffer: int = 60,
    ) -> None:
        self.paths = paths
        self.engine_factory = engine_factory
        self.cancel = cancel
        self.stop_event = threading.Event()
        self.on_detected = on_detected
        self.max_buffer = max_buffer
        self.results: dict[Path, tuple[list[DetectedFace] | None, str | None]] = {}
        self.condition = threading.Condition()
        self.finished = False
        self.fatal_error: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True, name="face-detection-prefetch")

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            engine = self.engine_factory()
            for path in self.paths:
                with self.condition:
                    self.condition.wait_for(
                        lambda: len(self.results) < self.max_buffer
                        or self.cancel.is_set()
                        or self.stop_event.is_set()
                    )
                if self.cancel.is_set() or self.stop_event.is_set():
                    break
                try:
                    faces = engine.detect_faces(path)
                    result = (faces, None)
                except Exception as exc:
                    faces = []
                    result = (None, str(exc))
                with self.condition:
                    self.results[path] = result
                    self.condition.notify_all()
                if self.on_detected and faces:
                    try:
                        self.on_detected(path, faces)
                    except Exception:
                        # UI preview notifications must not stop catalog detection.
                        pass
        except Exception as exc:
            self.fatal_error = str(exc)
        finally:
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def get(self, path: Path) -> list[DetectedFace]:
        with self.condition:
            self.condition.wait_for(
                lambda: path in self.results
                or self.finished
                or self.cancel.is_set()
                or self.stop_event.is_set()
            )
            if path not in self.results:
                if self.fatal_error:
                    raise RuntimeError(f"Background face detector failed: {self.fatal_error}")
                return []
            faces, error = self.results.pop(path)
            self.condition.notify_all()
        if error:
            raise ValueError(error)
        return faces or []

    def buffered_faces(self) -> list[tuple[Path, DetectedFace]]:
        with self.condition:
            return [
                (path, face)
                for path, (faces, _error) in self.results.items()
                for face in (faces or [])
            ]

    def stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
