from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
import truststore
from PIL import Image, ImageOps

NSFW_THRESHOLD = 0.45
NSFW_MODEL_VERSION = 3
VIT_MODEL_URL = (
    "https://huggingface.co/onnx-community/nsfw_image_detection-ONNX/resolve/main/"
    "onnx/model_int8.onnx"
)
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
    vit_score: float = 0.0
    review_required: bool = False


class NsfwDetector:
    """Local ONNX nudity detector with a single conservative gallery score."""

    def __init__(self, model_dir: Path | None = None) -> None:
        from nudenet import NudeDetector

        self.detector = NudeDetector()
        target_dir = model_dir or Path(__file__).resolve().parents[1] / "models"
        target_dir.mkdir(parents=True, exist_ok=True)
        model_path = target_dir / "falconsai-nsfw-vit-int8.onnx"
        if not model_path.is_file():
            temporary = model_path.with_suffix(".tmp")
            truststore.inject_into_ssl()
            with requests.get(VIT_MODEL_URL, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            temporary.replace(model_path)
        self.vit = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def _vit_score(self, path: Path) -> float:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").resize(
                (224, 224), Image.Resampling.BILINEAR
            )
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = ((pixels - 0.5) / 0.5).transpose(2, 0, 1)[None, ...]
        input_name = self.vit.get_inputs()[0].name
        logits = np.asarray(self.vit.run(None, {input_name: pixels})[0][0], dtype=np.float64)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return float(probabilities[1])

    def classify(self, path: Path) -> NsfwResult:
        detections = self.detector.detect(str(path))
        vit_score = self._vit_score(path)
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
        nude_score = max(
            (float(item["score"]) for item in summarized if bool(item["explicit"])), default=0.0
        )
        vit_positive = vit_score >= 0.65
        nude_positive = nude_score >= NSFW_THRESHOLD
        agreement = vit_positive and nude_positive
        review_required = vit_positive != nude_positive or 0.45 <= vit_score < 0.65
        combined = (
            {
                "label": "WHOLE_IMAGE_NSFW",
                "score": vit_score,
                "explicit": agreement,
                "model": "Falconsai ViT",
            },
            *summarized,
        )
        return NsfwResult(max(vit_score, nude_score) if agreement else 0.0, combined, vit_score, review_required)

    def score(self, path: Path) -> float:
        return self.classify(path).score
