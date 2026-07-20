from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SCREENSHOT_WORDS = ("screenshot", "screen shot", "screen_capture", "screen-capture")
DOCUMENT_WORDS = ("receipt", "document", "invoice", "statement", "paperwork", "scan_")


def filename_media_kind(path: Path) -> str | None:
    name = path.stem.casefold()
    if any(word in name for word in SCREENSHOT_WORDS):
        return "screenshot"
    if any(word in name for word in DOCUMENT_WORDS):
        return "document"
    return None


def classify_media_kind(path: Path) -> str:
    """Classify screenshots and document-like photos with conservative local heuristics."""
    named = filename_media_kind(path)
    if named:
        return named
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return "photo"
    height, width = image.shape[:2]
    scale = min(1.0, 900 / max(height, width))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_neutral = np.mean((hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 175))
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(np.mean(edges > 0))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    large_page = False
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) == 4 and cv2.contourArea(contour) >= image_area * 0.32:
            large_page = True
            break
    if bright_neutral >= 0.42 and 0.015 <= edge_density <= 0.20 and large_page:
        return "document"
    return "photo"
