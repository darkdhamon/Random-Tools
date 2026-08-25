from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import subprocess
import sys
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

AGE_MODEL = {
    "age_googlenet.onnx": (
        "https://huggingface.co/onnxmodelzoo/age_googlenet/resolve/main/"
        "age_googlenet.onnx?download=true"
    ),
}

ART_MODELS = {
    "clip-vit-base-patch32-q4.onnx": (
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
        "onnx/model_q4.onnx?download=true"
    ),
    "clip-tokenizer.json": (
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
        "tokenizer.json?download=true"
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


def ensure_age_model(
    folder: Path, status: Callable[[str], None] | None = None
) -> Path:
    """Download the Apache-licensed ONNX Model Zoo age classifier on first use."""
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, url in AGE_MODEL.items():
        destination = folder / filename
        if not destination.exists() or destination.stat().st_size < 1_000:
            if status:
                status(f"Downloading {filename}…")
            temporary = destination.with_suffix(destination.suffix + ".download")
            request = Request(url, headers={"User-Agent": "FacePhotoFinder/1.0"})
            try:
                with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            except OSError:
                if sys.platform != "win32":
                    raise
                # Some Windows installations cannot perform a certificate revocation
                # check for Intel's CDN. curl still validates the certificate while
                # skipping only that unavailable revocation lookup.
                subprocess.run(
                    [
                        "curl.exe", "--location", "--fail", "--silent", "--show-error",
                        "--ssl-no-revoke", "--output", str(temporary), url,
                    ],
                    check=True,
                )
            temporary.replace(destination)
        paths.append(destination)
    return paths[0]


def ensure_art_models(
    folder: Path, status: Callable[[str], None] | None = None
) -> tuple[Path, Path]:
    """Download the quantized local CLIP artwork classifier on first use."""
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, url in ART_MODELS.items():
        destination = folder / filename
        minimum_size = 10_000_000 if destination.suffix == ".onnx" else 100_000
        if not destination.is_file() or destination.stat().st_size < minimum_size:
            if status:
                status(f"Downloading artwork classifier {filename}…")
            temporary = destination.with_suffix(destination.suffix + ".download")
            subprocess.run(
                ["curl.exe", "--location", "--fail", "--silent", "--show-error",
                 "--ssl-no-revoke", "--output", str(temporary), url],
                check=True,
            )
            temporary.replace(destination)
        paths.append(destination)
    return paths[0], paths[1]
