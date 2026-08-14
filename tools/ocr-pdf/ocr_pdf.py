#!/usr/bin/env python3
"""Create a searchable PDF from one image-based PDF or a sequence of images."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS



def collect_images(inputs: Iterable[Path]) -> list[Path]:
    """Return image inputs in the exact order supplied, expanding folders by name."""
    images: list[Path] = []
    for item in inputs:
        if item.is_dir():
            images.extend(sorted((path for path in item.iterdir() if path.is_file() and is_image(path)), key=lambda p: p.name.casefold()))
        elif is_image(item):
            images.append(item)
        else:
            raise ValueError(f"Not an image file or folder: {item}")
    if not images:
        raise ValueError("No image files were found.")
    return images


def create_image_pdf(images: Iterable[Path], destination: Path) -> None:
    """Combine ordered images into a PDF without changing their visible orientation."""
    pages: list[Image.Image] = []
    try:
        for path in images:
            with Image.open(path) as source:
                page = ImageOps.exif_transpose(source).convert("RGB")
                pages.append(page.copy())
        pages[0].save(destination, "PDF", resolution=300.0, save_all=True, append_images=pages[1:])
    finally:
        for page in pages:
            page.close()


def run_ocr(input_pdf: Path, output_pdf: Path, language: str, deskew: bool, force_ocr: bool) -> None:
    command = [sys.executable, "-m", "ocrmypdf", "--output-type", "pdf", "--language", language]
    command.append("--force-ocr" if force_ocr else "--skip-text")
    if deskew:
        command.append("--deskew")
    command.extend([str(input_pdf), str(output_pdf)])
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError("Python could not start OCRmyPDF. Install this tool's requirements first.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "OCRmyPDF could not create the PDF. Confirm that Tesseract OCR and Ghostscript are installed and available on PATH."
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a searchable OCR PDF from a PDF, image files, or one folder of image files."
    )
    parser.add_argument("input", nargs="+", type=Path, help="One PDF, image files in page order, or one folder of images")
    parser.add_argument("output", type=Path, help="Destination searchable PDF")
    parser.add_argument("--language", default="eng", help="Tesseract language code(s), such as eng or eng+spa")
    parser.add_argument("--no-deskew", action="store_true", help="Do not straighten slightly tilted pages")
    parser.add_argument("--force-ocr", action="store_true", help="Re-OCR every page, including pages that already contain text")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = [path.expanduser().resolve() for path in args.input]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if len(inputs) == 1 and inputs[0].suffix.lower() == ".pdf":
        source_pdf = inputs[0]
        if not source_pdf.is_file():
            raise ValueError(f"PDF was not found: {source_pdf}")
        run_ocr(source_pdf, output, args.language, not args.no_deskew, args.force_ocr)
    else:
        images = collect_images(inputs)
        with tempfile.TemporaryDirectory(prefix="ocr-pdf-") as temporary_directory:
            source_pdf = Path(temporary_directory) / "images.pdf"
            create_image_pdf(images, source_pdf)
            run_ocr(source_pdf, output, args.language, not args.no_deskew, args.force_ocr)

    print(f"Created searchable PDF: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
