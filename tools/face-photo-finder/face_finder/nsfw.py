from __future__ import annotations

from pathlib import Path

NSFW_THRESHOLD = 0.45
EXPLICIT_CLASSES = {
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
}


class NsfwDetector:
    """Local ONNX nudity detector with a single conservative gallery score."""

    def __init__(self) -> None:
        from nudenet import NudeDetector

        self.detector = NudeDetector()

    def score(self, path: Path) -> float:
        detections = self.detector.detect(str(path))
        return max(
            (float(item.get("score", 0.0)) for item in detections if item.get("class") in EXPLICIT_CLASSES),
            default=0.0,
        )
