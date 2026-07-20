from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

PROMPTS = (
    "an ordinary photograph of a real living person",
    "a painting, drawing, cartoon, or illustration of a person",
    "a statue, sculpture, mannequin, wax figure, or doll",
)
KINDS = ("photo", "artwork", "statue")
ART_MODEL_VERSION = 1
AUTOMATIC_EXCLUSION_THRESHOLD = 0.72
AUTOMATIC_EXCLUSION_MARGIN = 0.25


@dataclass(frozen=True)
class ArtClassification:
    kind: str
    confidence: float
    exclude_from_biometrics: bool


def classification_from_probabilities(probabilities: np.ndarray) -> ArtClassification:
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if scores.size != len(KINDS):
        raise ValueError(f"Expected {len(KINDS)} artwork scores, received {scores.size}.")
    winning = int(np.argmax(scores))
    confidence = float(scores[winning])
    exclude = (
        winning > 0
        and confidence >= AUTOMATIC_EXCLUSION_THRESHOLD
        and confidence - float(scores[0]) >= AUTOMATIC_EXCLUSION_MARGIN
    )
    return ArtClassification(KINDS[winning], confidence, exclude)


class ArtClassifier:
    """Local CLIP classifier for photographic people versus artwork and statues."""

    def __init__(self, model_path: Path, tokenizer_path: Path) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        encodings = tokenizer.encode_batch(list(PROMPTS))
        self.input_ids = np.full((len(encodings), 77), 49407, dtype=np.int64)
        self.attention_mask = np.zeros((len(encodings), 77), dtype=np.int64)
        for index, encoding in enumerate(encodings):
            length = min(77, len(encoding.ids))
            self.input_ids[index, :length] = encoding.ids[:length]
            self.attention_mask[index, :length] = 1

    def classify(self, image: np.ndarray) -> ArtClassification:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = 224.0 / min(height, width)
        resized = cv2.resize(
            rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_CUBIC
        )
        y = max(0, (resized.shape[0] - 224) // 2)
        x = max(0, (resized.shape[1] - 224) // 2)
        pixels = resized[y:y + 224, x:x + 224].astype(np.float32) / 255.0
        mean = np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
        std = np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)
        pixels = ((pixels - mean) / std).transpose(2, 0, 1)[None, ...]
        logits = np.asarray(self.session.run(
            ["logits_per_image"],
            {"input_ids": self.input_ids, "attention_mask": self.attention_mask,
             "pixel_values": pixels},
        )[0][0], dtype=np.float64)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return classification_from_probabilities(probabilities)


def face_context(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = bbox
    padding_x, padding_y = round(width * 0.75), round(height * 0.75)
    left, top = max(0, x - padding_x), max(0, y - padding_y)
    right = min(image.shape[1], x + width + padding_x)
    bottom = min(image.shape[0], y + height + padding_y)
    return image[top:bottom, left:right]
