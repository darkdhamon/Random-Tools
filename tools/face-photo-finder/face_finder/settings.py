from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    reference_paths: tuple[Path, ...] = ()
    search_folder: Path | None = None


def default_settings_path() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"])
    else:
        root = Path.home() / ".config"
    return root / "FacePhotoFinder" / "settings.json"


def load_settings(path: Path | None = None) -> AppSettings:
    settings_path = path or default_settings_path()
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return AppSettings()

    references = tuple(
        candidate
        for value in data.get("reference_paths", [])
        if isinstance(value, str) and (candidate := Path(value)).is_file()
    )
    folder_value = data.get("search_folder")
    folder = Path(folder_value) if isinstance(folder_value, str) else None
    if folder is not None and not folder.is_dir():
        folder = None
    return AppSettings(references, folder)


def save_settings(settings: AppSettings, path: Path | None = None) -> None:
    settings_path = path or default_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings_path.with_suffix(".tmp")
    data = {
        "reference_paths": [str(item) for item in settings.reference_paths],
        "search_folder": str(settings.search_folder) if settings.search_folder else None,
    }
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(settings_path)

