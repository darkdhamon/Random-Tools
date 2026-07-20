from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

NSFW_THRESHOLD = 0.45
EXPLICIT_CLASSES = {
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
}


@dataclass(frozen=True)
class NsfwResult:
    score: float
    detections: tuple[dict[str, object], ...]


class NsfwDetector:
    """Local ONNX nudity detector with a single conservative gallery score."""

    def __init__(self) -> None:
        from nudenet import NudeDetector

        self.detector = NudeDetector()

    def classify(self, path: Path) -> NsfwResult:
        detections = self.detector.detect(str(path))
        summarized = tuple(
            {
                "label": str(item.get("class", "UNKNOWN")),
                "score": float(item.get("score", 0.0)),
                "explicit": item.get("class") in EXPLICIT_CLASSES,
            }
            for item in sorted(
                detections, key=lambda value: float(value.get("score", 0.0)), reverse=True
            )
            if float(item.get("score", 0.0)) >= 0.20
        )
        score = max(
            (float(item["score"]) for item in summarized if bool(item["explicit"])), default=0.0
        )
        return NsfwResult(score, summarized)

    def score(self, path: Path) -> float:
        return self.classify(path).score
