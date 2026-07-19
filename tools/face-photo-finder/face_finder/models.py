from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from urllib.request import Request, urlopen

MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx"
    ),
}


def ensure_models(folder: Path, status: Callable[[str], None] | None = None) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, url in MODELS.items():
        destination = folder / filename
        if not destination.exists() or destination.stat().st_size < 10_000:
            if status:
                status(f"Downloading {filename}…")
            temporary = destination.with_suffix(destination.suffix + ".download")
            request = Request(url, headers={"User-Agent": "FacePhotoFinder/1.0"})
            with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(destination)
        paths.append(destination)
    return paths[0], paths[1]

