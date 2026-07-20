from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests
import truststore

MODEL_URL = (
    "https://huggingface.co/opencv/object_detection_nanodet/resolve/main/"
    "object_detection_nanodet_2022nov.onnx?download=true"
)
PET_CLASSES = {15: "Cat", 16: "Dog"}


@dataclass(frozen=True)
class PetDetection:
    species: str
    confidence: float
    bbox: tuple[float, float, float, float]


class PetDetector:
    """Apache-licensed OpenCV NanoDet COCO detector limited to cats and dogs."""

    def __init__(self, model_dir: Path | None = None, threshold: float = 0.35) -> None:
        folder = model_dir or Path(__file__).resolve().parents[1] / "models"
        folder.mkdir(parents=True, exist_ok=True)
        model = folder / "object_detection_nanodet_2022nov.onnx"
        if not model.is_file() or model.stat().st_size < 1_000_000:
            temporary = model.with_suffix(".download")
            truststore.inject_into_ssl()
            with requests.get(MODEL_URL, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            output.write(chunk)
            temporary.replace(model)
        self.net = cv2.dnn.readNet(str(model))
        self.threshold = threshold
        self.strides = (8, 16, 32, 64)
        self.project = np.arange(8)
        self.anchors = []
        for stride in self.strides:
            size = 416 // stride
            x, y = np.meshgrid(np.arange(size) * stride, np.arange(size) * stride)
            self.anchors.append(np.column_stack(
                ((x.flatten() + 0.5 * (stride - 1)), (y.flatten() + 0.5 * (stride - 1)))
            ))

    def detect(self, path: Path) -> list[PetDetection]:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read image: {path.name}")
        height, width = image.shape[:2]
        resized = cv2.resize(image, (416, 416)).astype(np.float32)
        mean = np.array([103.53, 116.28, 123.675], np.float32).reshape(1, 1, 3)
        std = np.array([57.375, 57.12, 58.395], np.float32).reshape(1, 1, 3)
        self.net.setInput(cv2.dnn.blobFromImage((resized - mean) / std))
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        boxes: list[list[float]] = []
        scores: list[float] = []
        classes: list[int] = []
        for stride, class_scores, distances, anchors in zip(
            self.strides, outputs[::2], outputs[1::2], self.anchors
        ):
            class_scores = np.squeeze(class_scores, axis=0) if class_scores.ndim == 3 else class_scores
            distances = np.squeeze(distances, axis=0) if distances.ndim == 3 else distances
            probabilities = np.exp(distances.reshape(-1, 8))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            decoded = np.dot(probabilities, self.project).reshape(-1, 4) * stride
            class_ids = np.argmax(class_scores, axis=1)
            confidences = np.max(class_scores, axis=1)
            for index in np.where(np.isin(class_ids, list(PET_CLASSES)) & (confidences >= self.threshold))[0]:
                x1 = max(0.0, anchors[index, 0] - decoded[index, 0])
                y1 = max(0.0, anchors[index, 1] - decoded[index, 1])
                x2 = min(416.0, anchors[index, 0] + decoded[index, 2])
                y2 = min(416.0, anchors[index, 1] + decoded[index, 3])
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(confidences[index]))
                classes.append(int(class_ids[index]))
        keep = cv2.dnn.NMSBoxes(boxes, scores, self.threshold, 0.6)
        detections = []
        for index in np.asarray(keep).reshape(-1) if len(keep) else []:
            x, y, box_width, box_height = boxes[int(index)]
            detections.append(PetDetection(
                PET_CLASSES[classes[int(index)]], scores[int(index)],
                (x / 416, y / 416, box_width / 416, box_height / 416),
            ))
        return detections
