from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

NSFW_THRESHOLD = 0.45
NSFW_MODEL_VERSION = 2
EXPLICIT_CLASSES = {
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
}
CLASS_THRESHOLDS = {
    "ANUS_EXPOSED": 0.45,
    "BUTTOCKS_EXPOSED": 0.65,
    "FEMALE_BREAST_EXPOSED": 0.70,
    "FEMALE_GENITALIA_EXPOSED": 0.45,
    "MALE_GENITALIA_EXPOSED": 0.45,
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
        strongest: dict[str, float] = {}
        for item in detections:
            label = str(item.get("class", "UNKNOWN"))
            strongest[label] = max(strongest.get(label, 0.0), float(item.get("score", 0.0)))
        male_face = strongest.get("FACE_MALE", 0.0)
        female_face = strongest.get("FACE_FEMALE", 0.0)

        def contributes(label: str, score: float) -> bool:
            threshold = CLASS_THRESHOLDS.get(label)
            if threshold is None:
                return False
            if label == "FEMALE_BREAST_EXPOSED" and male_face >= 0.60 and female_face < 0.40:
                threshold = max(threshold, 0.85)
            return score >= threshold

        summarized = tuple(
            {
                "label": label,
                "score": score,
                "explicit": contributes(label, score),
            }
            for label, score in sorted(strongest.items(), key=lambda value: value[1], reverse=True)
            if score >= 0.20
        )
        score = max(
            (float(item["score"]) for item in summarized if bool(item["explicit"])), default=0.0
        )
        return NsfwResult(score, summarized)

    def score(self, path: Path) -> float:
        return self.classify(path).score
