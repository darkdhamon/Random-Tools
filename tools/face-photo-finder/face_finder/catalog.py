from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import shutil
import sqlite3
import zipfile

import numpy as np
import cv2
from PIL import Image, ExifTags

PROFILE_MAX_SAMPLES = 64
PROFILE_DUPLICATE_SIMILARITY = 0.92


def image_gps_coordinates(path: Path) -> tuple[float, float] | None:
    """Read decimal GPS coordinates from EXIF without modifying the image."""
    try:
        with Image.open(path) as image:
            gps = image.getexif().get_ifd(ExifTags.IFD.GPSInfo)
        latitude_values, latitude_ref = gps.get(2), gps.get(1)
        longitude_values, longitude_ref = gps.get(4), gps.get(3)
        if not latitude_values or not longitude_values:
            return None

        def decimal(values: object, reference: object) -> float:
            parts = [float(value) for value in values]  # type: ignore[union-attr]
            result = parts[0] + parts[1] / 60 + parts[2] / 3600
            return -result if str(reference).upper() in ("S", "W") else result

        return decimal(latitude_values, latitude_ref), decimal(longitude_values, longitude_ref)
    except (OSError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


@dataclass(frozen=True)
class KnownIdentity:
    identity_id: int
    name: str
    embeddings: tuple[np.ndarray, ...]
    sample_years: tuple[int | None, ...] = ()
    birth_year: int | None = None


@dataclass(frozen=True)
class UnknownGroup:
    group_id: int
    embeddings: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class CatalogFace:
    face_id: int
    identity_id: int | None
    identity_name: str | None
    embedding: np.ndarray
    preview: np.ndarray | None
    intentionally_unknown: bool
    bbox: tuple[int, int, int, int] | None
    unknown_group_id: int | None
    sharpness: float | None
    profile_eligible: bool
    is_art: bool


@dataclass(frozen=True)
class CatalogImage:
    image_id: int
    path: Path
    size: int
    modified_ns: int
    face_count: int
    identified_count: int
    capture_year: int | None = None


@dataclass(frozen=True)
class IdentityAssignment:
    face_id: int
    image_path: Path
    preview: np.ndarray | None
    is_art: bool
    profile_eligible: bool
    capture_year: int | None = None
    estimated_age: float | None = None
    capture_year_overridden: bool = False
    missing_since: str | None = None


@dataclass(frozen=True)
class ReconcileResult:
    relocated: int = 0
    newly_missing: int = 0


class FaceCatalog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=10.0)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.execute("PRAGMA busy_timeout = 10000")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS identities (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                created_at TEXT NOT NULL,
                birth_year INTEGER
            );
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                size INTEGER NOT NULL,
                modified_ns INTEGER NOT NULL,
                face_count INTEGER NOT NULL,
                identified_count INTEGER NOT NULL,
                scanned_at TEXT NOT NULL,
                capture_year INTEGER,
                capture_year_override INTEGER,
                capture_date TEXT,
                content_hash TEXT,
                missing_since TEXT
            );
            CREATE TABLE IF NOT EXISTS unknown_groups (
                id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY,
                image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                face_index INTEGER NOT NULL,
                identity_id INTEGER REFERENCES identities(id) ON DELETE SET NULL,
                embedding BLOB NOT NULL,
                preview BLOB,
                intentionally_unknown INTEGER NOT NULL DEFAULT 0,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_width INTEGER,
                bbox_height INTEGER,
                unknown_group_id INTEGER REFERENCES unknown_groups(id) ON DELETE SET NULL,
                sharpness REAL,
                profile_eligible INTEGER NOT NULL DEFAULT 1,
                is_art INTEGER NOT NULL DEFAULT 0,
                estimated_age REAL,
                UNIQUE(image_id, face_index)
            );
            CREATE INDEX IF NOT EXISTS faces_identity_idx ON faces(identity_id);
            CREATE TABLE IF NOT EXISTS image_metadata (
                image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '',
                rating INTEGER NOT NULL DEFAULT 0 CHECK(rating BETWEEN 0 AND 5),
                nsfw_override INTEGER,
                media_kind_override TEXT,
                location_name TEXT NOT NULL DEFAULT '',
                latitude REAL,
                longitude REAL
            );
            """
        )
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(faces)")}
        identity_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(identities)")}
        image_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(images)")}
        if "birth_year" not in identity_columns:
            self.connection.execute("ALTER TABLE identities ADD COLUMN birth_year INTEGER")
        if "capture_year" not in image_columns:
            self.connection.execute("ALTER TABLE images ADD COLUMN capture_year INTEGER")
        if "capture_year_override" not in image_columns:
            self.connection.execute("ALTER TABLE images ADD COLUMN capture_year_override INTEGER")
        if "capture_date" not in image_columns:
            self.connection.execute("ALTER TABLE images ADD COLUMN capture_date TEXT")
        if "content_hash" not in image_columns:
            self.connection.execute("ALTER TABLE images ADD COLUMN content_hash TEXT")
        if "missing_since" not in image_columns:
            self.connection.execute("ALTER TABLE images ADD COLUMN missing_since TEXT")
        if "nsfw_score" not in image_columns:
            self.connection.execute("ALTER TABLE images ADD COLUMN nsfw_score REAL")
        if "nsfw_scanned_at" not in image_columns:
            self.connection.execute("ALTER TABLE images ADD COLUMN nsfw_scanned_at TEXT")
        if "nsfw_details" not in image_columns:
            self.connection.execute("ALTER TABLE images ADD COLUMN nsfw_details TEXT")
        if "nsfw_model_version" not in image_columns:
            self.connection.execute("ALTER TABLE images ADD COLUMN nsfw_model_version INTEGER")
        if "nsfw_vit_score" not in image_columns:
            self.connection.execute("ALTER TABLE images ADD COLUMN nsfw_vit_score REAL")
        if "nsfw_review_required" not in image_columns:
            self.connection.execute(
                "ALTER TABLE images ADD COLUMN nsfw_review_required INTEGER NOT NULL DEFAULT 0"
            )
        if "media_kind" not in image_columns:
            self.connection.execute("ALTER TABLE images ADD COLUMN media_kind TEXT")
        if "gps_latitude" not in image_columns:
            self.connection.execute("ALTER TABLE images ADD COLUMN gps_latitude REAL")
        if "gps_longitude" not in image_columns:
            self.connection.execute("ALTER TABLE images ADD COLUMN gps_longitude REAL")
        if "gps_scanned_at" not in image_columns:
            self.connection.execute("ALTER TABLE images ADD COLUMN gps_scanned_at TEXT")
        metadata_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(image_metadata)")}
        if "nsfw_override" not in metadata_columns:
            self.connection.execute("ALTER TABLE image_metadata ADD COLUMN nsfw_override INTEGER")
        if "media_kind_override" not in metadata_columns:
            self.connection.execute("ALTER TABLE image_metadata ADD COLUMN media_kind_override TEXT")
        if "location_name" not in metadata_columns:
            self.connection.execute("ALTER TABLE image_metadata ADD COLUMN location_name TEXT NOT NULL DEFAULT ''")
        if "latitude" not in metadata_columns:
            self.connection.execute("ALTER TABLE image_metadata ADD COLUMN latitude REAL")
        if "longitude" not in metadata_columns:
            self.connection.execute("ALTER TABLE image_metadata ADD COLUMN longitude REAL")
        if "preview" not in columns:
            self.connection.execute("ALTER TABLE faces ADD COLUMN preview BLOB")
        if "intentionally_unknown" not in columns:
            self.connection.execute(
                "ALTER TABLE faces ADD COLUMN intentionally_unknown INTEGER NOT NULL DEFAULT 0"
            )
        for column in ("bbox_x", "bbox_y", "bbox_width", "bbox_height"):
            if column not in columns:
                self.connection.execute(f"ALTER TABLE faces ADD COLUMN {column} INTEGER")
        if "unknown_group_id" not in columns:
            self.connection.execute("ALTER TABLE faces ADD COLUMN unknown_group_id INTEGER")
        if "sharpness" not in columns:
            self.connection.execute("ALTER TABLE faces ADD COLUMN sharpness REAL")
        if "profile_eligible" not in columns:
            self.connection.execute(
                "ALTER TABLE faces ADD COLUMN profile_eligible INTEGER NOT NULL DEFAULT 1"
            )
            from .scanner import MIN_PROFILE_SHARPNESS, face_sharpness

            for face_id, preview in self.connection.execute(
                "SELECT id, preview FROM faces WHERE preview IS NOT NULL"
            ).fetchall():
                crop = cv2.imdecode(np.frombuffer(preview, dtype=np.uint8), cv2.IMREAD_COLOR)
                if crop is not None:
                    score = face_sharpness(crop)
                    self.connection.execute(
                        "UPDATE faces SET sharpness = ?, profile_eligible = ? WHERE id = ?",
                        (score, int(score >= MIN_PROFILE_SHARPNESS), face_id),
                    )
            self.connection.commit()
        if "is_art" not in columns:
            self.connection.execute("ALTER TABLE faces ADD COLUMN is_art INTEGER NOT NULL DEFAULT 0")
        if "estimated_age" not in columns:
            self.connection.execute("ALTER TABLE faces ADD COLUMN estimated_age REAL")
        legacy_unknowns = self.connection.execute(
            "SELECT id FROM faces WHERE intentionally_unknown = 1 AND unknown_group_id IS NULL"
        ).fetchall()
        with self.connection:
            for (face_id,) in legacy_unknowns:
                cursor = self.connection.execute(
                    "INSERT INTO unknown_groups(created_at) VALUES (?)",
                    (datetime.now(timezone.utc).isoformat(),),
                )
                self.connection.execute(
                    "UPDATE faces SET unknown_group_id = ? WHERE id = ?", (cursor.lastrowid, face_id)
                )

    def close(self) -> None:
        self.connection.close()

    def reset(self) -> None:
        """Permanently clear all cataloged biometric and scan data."""
        with self.connection:
            self.connection.execute("DELETE FROM faces")
            self.connection.execute("DELETE FROM images")
            self.connection.execute("DELETE FROM identities")
            self.connection.execute("DELETE FROM unknown_groups")
        self.connection.execute("VACUUM")

    def gallery_photos(
        self, search: str = "", identity_id: int | None = None, year: int | None = None,
        limit: int = 100, offset: int = 0, nsfw_filter: str = "all",
        excluded_kinds: tuple[str, ...] = (), media_kind: str | None = None,
    ) -> list[dict[str, object]]:
        clauses = ["images.missing_since IS NULL"]
        values: list[object] = []
        if search:
            clauses.append("(images.path LIKE ? OR metadata.title LIKE ? OR metadata.tags LIKE ?)")
            pattern = f"%{search}%"
            values.extend((pattern, pattern, pattern))
        if identity_id is not None:
            clauses.append("EXISTS (SELECT 1 FROM faces WHERE faces.image_id = images.id AND faces.identity_id = ?)")
            values.append(identity_id)
        if year is not None:
            clauses.append("COALESCE(images.capture_year_override, images.capture_year) = ?")
            values.append(year)
        effective_nsfw = "COALESCE(metadata.nsfw_override, images.nsfw_score >= 0.45, 0)"
        if nsfw_filter == "safe":
            clauses.append(f"{effective_nsfw} = 0")
        elif nsfw_filter == "nsfw":
            clauses.append(f"{effective_nsfw} = 1")
        elif nsfw_filter in {"review", "conflict"}:
            clauses.append("images.nsfw_review_required = 1 AND metadata.nsfw_override IS NULL")
        effective_kind = "COALESCE(metadata.media_kind_override, images.media_kind, 'photo')"
        effective_date = """CASE WHEN images.capture_year_override IS NULL
                              OR CAST(substr(images.capture_date, 1, 4) AS INTEGER) = images.capture_year_override
                            THEN NULLIF(images.capture_date, '') END"""
        if media_kind:
            clauses.append(f"{effective_kind} = ?")
            values.append(media_kind)
        elif excluded_kinds:
            placeholders = ",".join("?" for _ in excluded_kinds)
            clauses.append(f"{effective_kind} NOT IN ({placeholders})")
            values.extend(excluded_kinds)
        values.extend((max(1, min(limit, 250)), max(0, offset)))
        rows = self.connection.execute(
            f"""SELECT images.id, images.path, images.face_count, images.identified_count,
                       COALESCE(images.capture_year_override, images.capture_year),
                       COALESCE(metadata.title, ''), COALESCE(metadata.description, ''),
                       COALESCE(metadata.tags, ''), COALESCE(metadata.rating, 0),
                       images.nsfw_score, metadata.nsfw_override,
                       COALESCE(metadata.nsfw_override, images.nsfw_score >= 0.45, 0),
                       COALESCE(metadata.media_kind_override, images.media_kind, 'photo'),
                       metadata.media_kind_override, COALESCE(metadata.location_name, ''),
                       COALESCE(metadata.latitude, images.gps_latitude),
                       COALESCE(metadata.longitude, images.gps_longitude), images.nsfw_details,
                       images.nsfw_vit_score, images.nsfw_review_required,
                       images.capture_year_override, {effective_date}
                FROM images LEFT JOIN image_metadata metadata ON metadata.image_id = images.id
                WHERE {' AND '.join(clauses)}
                ORDER BY COALESCE(images.capture_year_override, images.capture_year) DESC,
                         {effective_date} IS NULL, {effective_date} DESC, images.path DESC
                LIMIT ? OFFSET ?""",
            values,
        ).fetchall()
        photos = []
        for row in rows:
            capture_date = row[21]
            photos.append({
                "id": int(row[0]), "path": row[1], "name": Path(row[1]).name,
                "face_count": int(row[2]), "identified_count": int(row[3]), "year": row[4],
                "capture_date": capture_date,
                "title": row[5], "description": row[6], "tags": row[7], "rating": int(row[8]),
                "nsfw_score": row[9], "nsfw_override": row[10], "is_nsfw": bool(row[11]),
                "media_kind": row[12], "media_kind_override": row[13],
                "location_name": row[14], "latitude": row[15], "longitude": row[16],
                "nsfw_detections": json.loads(row[17]) if row[17] else [],
                "nsfw_vit_score": row[18], "nsfw_review_required": bool(row[19]),
            })
        return photos

    def gallery_photo(self, image_id: int) -> dict[str, object] | None:
        row = self.connection.execute(
            """SELECT images.path, images.face_count, images.identified_count,
                      COALESCE(images.capture_year_override, images.capture_year),
                      COALESCE(metadata.title, ''), COALESCE(metadata.description, ''),
                      COALESCE(metadata.tags, ''), COALESCE(metadata.rating, 0),
                      images.nsfw_score, metadata.nsfw_override,
                      COALESCE(metadata.nsfw_override, images.nsfw_score >= 0.45, 0),
                      COALESCE(metadata.media_kind_override, images.media_kind, 'photo'),
                      metadata.media_kind_override, COALESCE(metadata.location_name, ''),
                      COALESCE(metadata.latitude, images.gps_latitude),
                      COALESCE(metadata.longitude, images.gps_longitude), images.nsfw_details,
                      images.nsfw_vit_score, images.nsfw_review_required,
                      images.capture_year_override,
                      CASE WHEN images.capture_year_override IS NULL
                              OR CAST(substr(images.capture_date, 1, 4) AS INTEGER) = images.capture_year_override
                           THEN NULLIF(images.capture_date, '') END
               FROM images LEFT JOIN image_metadata metadata ON metadata.image_id = images.id
               WHERE images.id = ? AND images.missing_since IS NULL""",
            (image_id,),
        ).fetchone()
        if row is None:
            return None
        capture_date = row[20]
        photo: dict[str, object] = {
            "id": image_id, "path": row[0], "name": Path(row[0]).name,
            "face_count": int(row[1]), "identified_count": int(row[2]), "year": row[3],
            "title": row[4], "description": row[5], "tags": row[6], "rating": int(row[7]),
            "nsfw_score": row[8], "nsfw_override": row[9], "is_nsfw": bool(row[10]),
            "media_kind": row[11], "media_kind_override": row[12],
            "location_name": row[13], "latitude": row[14], "longitude": row[15],
            "nsfw_detections": json.loads(row[16]) if row[16] else [],
            "nsfw_vit_score": row[17], "nsfw_review_required": bool(row[18]),
            "capture_date": capture_date,
        }
        faces = self.connection.execute(
            """SELECT faces.id, identities.name, faces.intentionally_unknown, faces.is_art,
                      faces.estimated_age, faces.unknown_group_id
               FROM faces LEFT JOIN identities ON identities.id = faces.identity_id
               WHERE faces.image_id = ? ORDER BY faces.face_index""",
            (image_id,),
        ).fetchall()
        photo["faces"] = [
            {"id": int(row[0]), "name": row[1], "unknown": bool(row[2]), "art": bool(row[3]),
             "estimated_age": row[4], "unknown_group_id": row[5]}
            for row in faces
        ]
        return photo

    def image_path(self, image_id: int) -> Path | None:
        row = self.connection.execute(
            "SELECT path FROM images WHERE id = ? AND missing_since IS NULL", (image_id,)
        ).fetchone()
        path = Path(row[0]) if row else None
        return path if path and path.is_file() else None

    def backfill_capture_dates(self) -> tuple[int, int]:
        """Persist capture dates for legacy rows so timeline ordering is stable and fast."""
        rows = self.connection.execute(
            "SELECT id, path FROM images WHERE capture_date IS NULL AND missing_since IS NULL"
        ).fetchall()
        dated = 0
        with self.connection:
            for image_id, path_value in rows:
                capture_date = image_capture_date(Path(path_value))
                self.connection.execute(
                    "UPDATE images SET capture_date = ? WHERE id = ?",
                    (capture_date or "", image_id),
                )
                dated += capture_date is not None
        return len(rows), dated

    def delete_photo(self, image_id: int) -> Path:
        """Permanently delete a source photo and its cascading catalog records."""
        row = self.connection.execute(
            "SELECT path FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Photo was not found in the catalog.")
        path = Path(row[0])
        if not path.is_file():
            raise FileNotFoundError(f"Photo no longer exists: {path}")
        with self.connection:
            path.unlink()
            self.connection.execute("DELETE FROM images WHERE id = ?", (image_id,))
        return path

    def delete_photos(self, image_ids: list[int]) -> list[Path]:
        """Permanently delete several source photos and their catalog records."""
        unique_ids = list(dict.fromkeys(int(value) for value in image_ids))
        if not unique_ids:
            raise ValueError("Select at least one photo.")
        placeholders = ",".join("?" for _ in unique_ids)
        rows = self.connection.execute(
            f"SELECT id, path FROM images WHERE id IN ({placeholders})", unique_ids
        ).fetchall()
        found = {int(image_id): Path(path) for image_id, path in rows}
        missing_ids = [image_id for image_id in unique_ids if image_id not in found]
        if missing_ids:
            raise ValueError("One or more selected photos were not found in the catalog.")
        missing_files = [path for path in found.values() if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(f"Photo no longer exists: {missing_files[0]}")
        paths = [found[image_id] for image_id in unique_ids]
        with self.connection:
            for path in paths:
                path.unlink()
            self.connection.execute(
                f"DELETE FROM images WHERE id IN ({placeholders})", unique_ids
            )
        return paths

    def archive_photos(
        self, image_ids: list[int], archive_path: Path, library_root: Path
    ) -> list[Path]:
        """Move selected photos into an atomically replaced ZIP archive."""
        unique_ids = list(dict.fromkeys(int(value) for value in image_ids))
        if not unique_ids:
            raise ValueError("Select at least one photo.")
        placeholders = ",".join("?" for _ in unique_ids)
        rows = self.connection.execute(
            f"SELECT id, path FROM images WHERE id IN ({placeholders})", unique_ids
        ).fetchall()
        found = {int(image_id): Path(path) for image_id, path in rows}
        if len(found) != len(unique_ids):
            raise ValueError("One or more selected photos were not found in the catalog.")
        paths = [found[image_id] for image_id in unique_ids]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Photo no longer exists: {missing[0]}")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive_path.with_name(f"{archive_path.name}.tmp")
        if temporary.exists():
            temporary.unlink()
        existing_names: set[str] = set()
        try:
            if archive_path.is_file():
                shutil.copy2(archive_path, temporary)
            mode = "a" if temporary.exists() else "w"
            with zipfile.ZipFile(temporary, mode, zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                existing_names.update(archive.namelist())
                for image_id, path in zip(unique_ids, paths):
                    try:
                        member = path.resolve().relative_to(library_root.resolve()).as_posix()
                    except ValueError:
                        member = f"Archived/{image_id}-{path.name}"
                    original = member
                    suffix = 2
                    while member in existing_names:
                        candidate = Path(original)
                        member = str(
                            candidate.with_name(f"{candidate.stem}-{suffix}{candidate.suffix}")
                        ).replace("\\", "/")
                        suffix += 1
                    archive.write(path, member)
                    existing_names.add(member)
            temporary.replace(archive_path)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        with self.connection:
            for path in paths:
                path.unlink()
            self.connection.execute(
                f"DELETE FROM images WHERE id IN ({placeholders})", unique_ids
            )
        return paths

    def update_gallery_metadata(
        self, image_id: int, title: str, description: str, tags: str, rating: int,
        capture_year: int | None, nsfw_override: int | None = None,
        media_kind_override: str | None = None,
        location_name: str = "", latitude: float | None = None, longitude: float | None = None,
    ) -> None:
        if rating not in range(6):
            raise ValueError("Rating must be between 0 and 5.")
        if capture_year is not None and not 1900 <= capture_year <= datetime.now().year:
            raise ValueError("Capture year must be between 1900 and the current year.")
        if nsfw_override not in (None, 0, 1):
            raise ValueError("NSFW override must be automatic, safe, or NSFW.")
        if media_kind_override not in (None, "photo", "screenshot", "document"):
            raise ValueError("Content type must be automatic, photo, screenshot, or document.")
        if latitude is not None and not -90 <= latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90.")
        if longitude is not None and not -180 <= longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180.")
        with self.connection:
            self.connection.execute(
                """INSERT INTO image_metadata(image_id, title, description, tags, rating, nsfw_override,
                                               media_kind_override)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(image_id) DO UPDATE SET title=excluded.title,
                       description=excluded.description, tags=excluded.tags, rating=excluded.rating,
                       nsfw_override=excluded.nsfw_override,
                       media_kind_override=excluded.media_kind_override""",
                (image_id, title.strip(), description.strip(), tags.strip(), rating, nsfw_override,
                 media_kind_override),
            )
            self.connection.execute(
                """UPDATE image_metadata SET location_name = ?, latitude = ?, longitude = ?
                   WHERE image_id = ?""",
                (location_name.strip(), latitude, longitude, image_id),
            )
            self.connection.execute(
                "UPDATE images SET capture_year_override = ? WHERE id = ?", (capture_year, image_id)
            )

    def nsfw_score_for_path(self, path: Path) -> float | None:
        row = self.connection.execute(
            "SELECT nsfw_score FROM images WHERE path = ?", (str(path.resolve()),)
        ).fetchone()
        return row[0] if row else None

    def nsfw_classification_for_path(
        self, path: Path
    ) -> tuple[float | None, list[dict[str, object]] | None]:
        row = self.connection.execute(
            "SELECT nsfw_score, nsfw_details, nsfw_model_version FROM images WHERE path = ?",
            (str(path.resolve()),),
        ).fetchone()
        if row is None:
            return None, None
        from .nsfw import NSFW_MODEL_VERSION

        if row[2] != NSFW_MODEL_VERSION:
            return row[0], None
        return row[0], json.loads(row[1]) if row[1] is not None else None

    def set_nsfw_score(self, image_id: int, score: float) -> None:
        self.set_nsfw_classification(image_id, score, [], 0.0, False)

    def set_nsfw_classification(
        self, image_id: int, score: float, detections: list[dict[str, object]],
        vit_score: float = 0.0, review_required: bool = False,
    ) -> None:
        if not 0 <= score <= 1:
            raise ValueError("NSFW score must be between 0 and 1.")
        with self.connection:
            from .nsfw import NSFW_MODEL_VERSION

            self.connection.execute(
                """UPDATE images SET nsfw_score = ?, nsfw_details = ?, nsfw_scanned_at = ?,
                          nsfw_model_version = ?, nsfw_vit_score = ?, nsfw_review_required = ?
                   WHERE id = ?""",
                (
                    score, json.dumps(detections), datetime.now(timezone.utc).isoformat(),
                    NSFW_MODEL_VERSION, vit_score, int(review_required), image_id,
                ),
            )

    def media_kind_for_path(self, path: Path) -> str | None:
        row = self.connection.execute(
            "SELECT media_kind FROM images WHERE path = ?", (str(path.resolve()),)
        ).fetchone()
        return row[0] if row else None

    def set_media_kind(self, image_id: int, media_kind: str) -> None:
        if media_kind not in ("photo", "screenshot", "document"):
            raise ValueError("Invalid content type.")
        with self.connection:
            self.connection.execute("UPDATE images SET media_kind = ? WHERE id = ?", (media_kind, image_id))

    def ensure_image_location(self, image_id: int, path: Path) -> tuple[float, float] | None:
        row = self.connection.execute(
            "SELECT gps_latitude, gps_longitude, gps_scanned_at FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        if row is None:
            return None
        if row[2] is not None:
            return (float(row[0]), float(row[1])) if row[0] is not None and row[1] is not None else None
        coordinates = image_gps_coordinates(path)
        latitude, longitude = coordinates if coordinates else (None, None)
        with self.connection:
            self.connection.execute(
                "UPDATE images SET gps_latitude = ?, gps_longitude = ?, gps_scanned_at = ? WHERE id = ?",
                (latitude, longitude, datetime.now(timezone.utc).isoformat(), image_id),
            )
        return coordinates

    def identities(self) -> list[KnownIdentity]:
        rows = self.connection.execute(
            """SELECT identities.id, identities.name, identities.birth_year,
                      faces.embedding, COALESCE(images.capture_year_override, images.capture_year)
               FROM identities LEFT JOIN faces ON faces.identity_id = identities.id
                    AND faces.profile_eligible = 1
               LEFT JOIN images ON images.id = faces.image_id
               ORDER BY identities.name, faces.id"""
        ).fetchall()
        grouped: dict[tuple[int, str, int | None], list[tuple[np.ndarray, int | None]]] = {}
        for identity_id, name, birth_year, blob, capture_year in rows:
            grouped.setdefault((identity_id, name, birth_year), [])
            if blob is not None:
                grouped[(identity_id, name, birth_year)].append(
                    (np.frombuffer(blob, dtype=np.float32).copy(), capture_year)
                )
        identities: list[KnownIdentity] = []
        for key, values in grouped.items():
            samples = bounded_profile_samples(values)
            identities.append(KnownIdentity(key[0], key[1], tuple(v[0] for v in samples),
                                            tuple(v[1] for v in samples), key[2]))
        return identities

    def set_identity_birth_year(self, identity_id: int, birth_year: int | None) -> None:
        if birth_year is not None and not 1900 <= birth_year <= datetime.now().year:
            raise ValueError("Birth year must be between 1900 and the current year.")
        with self.connection:
            self.connection.execute(
                "UPDATE identities SET birth_year = ? WHERE id = ?", (birth_year, identity_id)
            )

    def unknown_groups(self) -> list[UnknownGroup]:
        rows = self.connection.execute(
            """SELECT unknown_groups.id, faces.embedding
               FROM unknown_groups LEFT JOIN faces ON faces.unknown_group_id = unknown_groups.id
                    AND faces.profile_eligible = 1
               ORDER BY unknown_groups.id, faces.id"""
        ).fetchall()
        grouped: dict[int, list[np.ndarray]] = {}
        for group_id, blob in rows:
            grouped.setdefault(int(group_id), [])
            if blob is not None:
                grouped[int(group_id)].append(np.frombuffer(blob, dtype=np.float32).copy())
        return [
            UnknownGroup(group_id, bounded_profile(values)) for group_id, values in grouped.items()
        ]

    def create_unknown_group(self) -> int:
        cursor = self.connection.execute(
            "INSERT INTO unknown_groups(created_at) VALUES (?)", (datetime.now(timezone.utc).isoformat(),)
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def unknown_group_face_count(self, group_id: int) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM faces WHERE unknown_group_id = ?", (group_id,)
        ).fetchone()
        return int(row[0])

    def assign_unknown_group(self, group_id: int, identity_id: int) -> tuple[int, int]:
        """Retroactively attach every face in an anonymous group to a named identity."""
        image_rows = self.connection.execute(
            "SELECT DISTINCT image_id FROM faces WHERE unknown_group_id = ?", (group_id,)
        ).fetchall()
        image_ids = [int(row[0]) for row in image_rows]
        with self.connection:
            cursor = self.connection.execute(
                """UPDATE faces SET identity_id = ?, intentionally_unknown = 0,
                          unknown_group_id = NULL WHERE unknown_group_id = ?""",
                (identity_id, group_id),
            )
            for image_id in image_ids:
                self.connection.execute(
                    """UPDATE images SET identified_count =
                       (SELECT COUNT(*) FROM faces WHERE faces.image_id = images.id
                        AND identity_id IS NOT NULL) WHERE id = ?""",
                    (image_id,),
                )
            self.connection.execute(
                """DELETE FROM unknown_groups WHERE id = ?
                   AND NOT EXISTS (SELECT 1 FROM faces WHERE unknown_group_id = ?)""",
                (group_id, group_id),
            )
        return int(cursor.rowcount), len(image_ids)

    def get_or_create_identity(self, name: str) -> int:
        clean_name = " ".join(name.split())
        if not clean_name:
            raise ValueError("An identity name is required.")
        existing = self.connection.execute("SELECT id FROM identities WHERE name = ?", (clean_name,)).fetchone()
        if existing:
            return int(existing[0])
        cursor = self.connection.execute(
            "INSERT INTO identities(name, created_at) VALUES (?, ?)",
            (clean_name, datetime.now(timezone.utc).isoformat()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def identity_assignments(self, identity_id: int) -> list[IdentityAssignment]:
        rows = self.connection.execute(
            """SELECT faces.id, images.path, faces.preview, faces.is_art, faces.profile_eligible,
                      COALESCE(images.capture_year_override, images.capture_year), faces.estimated_age,
                      images.capture_year_override IS NOT NULL, images.missing_since
               FROM faces JOIN images ON images.id = faces.image_id
               WHERE faces.identity_id = ? ORDER BY images.path, faces.face_index""",
            (identity_id,),
        ).fetchall()
        return [
            IdentityAssignment(
                int(row[0]),
                Path(row[1]),
                np.frombuffer(row[2], dtype=np.uint8).copy() if row[2] is not None else None,
                bool(row[3]),
                bool(row[4]),
                row[5],
                row[6],
                bool(row[7]),
                row[8],
            )
            for row in rows
        ]

    def reconcile_files(self, paths: list[Path]) -> ReconcileResult:
        """Relink moved files conservatively and mark truly absent catalog paths missing."""
        candidates: dict[int, list[Path]] = {}
        for path in paths:
            try:
                resolved = path.resolve()
                candidates.setdefault(resolved.stat().st_size, []).append(resolved)
            except OSError:
                continue
        rows = self.connection.execute(
            "SELECT id, path, size, content_hash, missing_since FROM images"
        ).fetchall()
        catalog_paths = {Path(row[1]) for row in rows}
        hash_cache: dict[Path, str] = {}

        def candidate_hash(path: Path) -> str:
            if path not in hash_cache:
                hash_cache[path] = file_content_hash(path)
            return hash_cache[path]

        relocated = 0
        newly_missing = 0
        now = datetime.now(timezone.utc).isoformat()
        with self.connection:
            for image_id, stored_path, size, stored_hash, missing_since in rows:
                old_path = Path(stored_path)
                if old_path.exists():
                    if missing_since is not None:
                        self.connection.execute(
                            "UPDATE images SET missing_since = NULL WHERE id = ?", (image_id,)
                        )
                    continue
                possible = [path for path in candidates.get(int(size), []) if path not in catalog_paths]
                match: Path | None = None
                if stored_hash:
                    matching: list[Path] = []
                    for path in possible:
                        digest = candidate_hash(path)
                        if digest == stored_hash:
                            matching.append(path)
                    if len(matching) == 1:
                        match = matching[0]
                else:
                    # Legacy rows predate fingerprints. Filename plus exact byte size is
                    # intentionally conservative; ambiguous candidates remain missing.
                    same_name = [path for path in possible if path.name.casefold() == old_path.name.casefold()]
                    if len(same_name) == 1:
                        match = same_name[0]
                if match is not None:
                    stat = match.stat()
                    digest = candidate_hash(match)
                    self.connection.execute(
                        """UPDATE images SET path = ?, size = ?, modified_ns = ?, content_hash = ?,
                                  missing_since = NULL WHERE id = ?""",
                        (str(match), stat.st_size, stat.st_mtime_ns, digest, image_id),
                    )
                    catalog_paths.add(match)
                    relocated += 1
                elif missing_since is None:
                    self.connection.execute(
                        "UPDATE images SET missing_since = ? WHERE id = ?", (now, image_id)
                    )
                    newly_missing += 1
        return ReconcileResult(relocated, newly_missing)

    def refresh_missing_status(self) -> int:
        """Refresh missing markers without attempting relocation discovery."""
        return self.reconcile_files([]).newly_missing

    def prune_missing_images(self) -> int:
        """Delete catalog rows whose source paths are still absent."""
        rows = self.connection.execute(
            "SELECT id, path FROM images WHERE missing_since IS NOT NULL"
        ).fetchall()
        image_ids = [int(image_id) for image_id, path in rows if not Path(path).exists()]
        if not image_ids:
            return 0
        placeholders = ",".join("?" for _ in image_ids)
        with self.connection:
            cursor = self.connection.execute(
                f"DELETE FROM images WHERE id IN ({placeholders})", image_ids
            )
        return int(cursor.rowcount)

    def set_face_estimated_age(self, face_id: int, estimated_age: float | None) -> None:
        if estimated_age is not None and not 0 <= estimated_age <= 100:
            raise ValueError("Estimated age must be between 0 and 100.")
        with self.connection:
            self.connection.execute(
                "UPDATE faces SET estimated_age = ? WHERE id = ?", (estimated_age, face_id)
            )

    def set_capture_year_for_faces(self, face_ids: list[int], capture_year: int | None) -> int:
        """Set or clear a persistent photo-year override for the selected face rows."""
        if capture_year is not None and not 1900 <= capture_year <= datetime.now().year:
            raise ValueError("Capture year must be between 1900 and the current year.")
        unique_ids = sorted(set(face_ids))
        if not unique_ids:
            return 0
        placeholders = ",".join("?" for _ in unique_ids)
        image_rows = self.connection.execute(
            f"SELECT DISTINCT image_id FROM faces WHERE id IN ({placeholders})", unique_ids
        ).fetchall()
        image_ids = [int(row[0]) for row in image_rows]
        if not image_ids:
            return 0
        image_placeholders = ",".join("?" for _ in image_ids)
        with self.connection:
            self.connection.execute(
                f"UPDATE images SET capture_year_override = ? WHERE id IN ({image_placeholders})",
                (capture_year, *image_ids),
            )
        return len(image_ids)

    def reassign_faces(self, face_ids: list[int], target_identity_id: int) -> int:
        unique_ids = sorted(set(face_ids))
        if not unique_ids:
            return 0
        placeholders = ",".join("?" for _ in unique_ids)
        with self.connection:
            cursor = self.connection.execute(
                f"""UPDATE faces SET identity_id = ?, intentionally_unknown = 0,
                    unknown_group_id = NULL WHERE id IN ({placeholders})""",
                (target_identity_id, *unique_ids),
            )
        return int(cursor.rowcount)

    def remove_identity_if_unused(self, identity_id: int) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                """DELETE FROM identities WHERE id = ?
                   AND NOT EXISTS (SELECT 1 FROM faces WHERE identity_id = identities.id)""",
                (identity_id,),
            )
        return cursor.rowcount > 0

    def cached_image(self, path: Path) -> CatalogImage | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        row = self.connection.execute(
            """SELECT id, path, size, modified_ns, face_count, identified_count,
                      COALESCE(capture_year_override, capture_year)
               FROM images WHERE path = ?""",
            (str(path.resolve()),),
        ).fetchone()
        if not row or row[2] != stat.st_size or row[3] != stat.st_mtime_ns:
            return None
        missing_geometry = self.connection.execute(
            """SELECT 1 FROM faces WHERE image_id = ? AND bbox_x IS NULL LIMIT 1""", (row[0],)
        ).fetchone()
        if missing_geometry:
            return None
        capture_year = row[6]
        if capture_year is None:
            # Enrich legacy rows lazily during the background scan. Doing this for the
            # whole library when a UI dialog opens can make Tk appear frozen.
            capture_year = image_capture_year(path)
            if capture_year is not None:
                with self.connection:
                    self.connection.execute(
                        "UPDATE images SET capture_year = ? WHERE id = ?", (capture_year, row[0])
                    )
        return CatalogImage(int(row[0]), Path(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), capture_year)

    def store_scan(
        self,
        path: Path,
        embeddings: list[np.ndarray],
        identities: list[int | None],
        previews: list[np.ndarray] | None = None,
        intentionally_unknown: list[bool] | None = None,
        boxes: list[tuple[int, int, int, int] | None] | None = None,
        unknown_group_ids: list[int | None] | None = None,
        sharpness_scores: list[float] | None = None,
        profile_eligible: list[bool] | None = None,
        is_art: list[bool] | None = None,
    ) -> CatalogImage:
        if len(embeddings) != len(identities):
            raise ValueError("Every face must have an identity assignment.")
        if previews is not None and len(previews) != len(embeddings):
            raise ValueError("Every face must have a corresponding preview.")
        if intentionally_unknown is None:
            intentionally_unknown = [False] * len(embeddings)
        if len(intentionally_unknown) != len(embeddings):
            raise ValueError("Every face must have an unknown-person status.")
        if boxes is None:
            boxes = [(0, 0, 0, 0)] * len(embeddings)
        if len(boxes) != len(embeddings):
            raise ValueError("Every face must have a bounding-box value.")
        if unknown_group_ids is None:
            unknown_group_ids = [None] * len(embeddings)
        if len(unknown_group_ids) != len(embeddings):
            raise ValueError("Every face must have an anonymous identity-group value.")
        if sharpness_scores is None:
            sharpness_scores = [0.0] * len(embeddings)
        if profile_eligible is None:
            profile_eligible = [True] * len(embeddings)
        if len(sharpness_scores) != len(embeddings) or len(profile_eligible) != len(embeddings):
            raise ValueError("Every face must have biometric profile-quality values.")
        if is_art is None:
            is_art = [False] * len(embeddings)
        if len(is_art) != len(embeddings):
            raise ValueError("Every face must have an artwork status.")
        resolved = path.resolve()
        stat = resolved.stat()
        capture_year = image_capture_year(resolved)
        capture_date = image_capture_date(resolved)
        identified_count = sum(value is not None for value in identities)
        with self.connection:
            self.connection.execute(
                """INSERT INTO images(path, size, modified_ns, face_count, identified_count, scanned_at,
                                      capture_year, capture_date, content_hash, missing_since)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                   ON CONFLICT(path) DO UPDATE SET size=excluded.size, modified_ns=excluded.modified_ns,
                   face_count=excluded.face_count, identified_count=excluded.identified_count,
                   scanned_at=excluded.scanned_at, capture_year=excluded.capture_year,
                   capture_date=excluded.capture_date,
                   content_hash=excluded.content_hash, missing_since=NULL""",
                (
                    str(resolved), stat.st_size, stat.st_mtime_ns, len(embeddings), identified_count,
                    datetime.now(timezone.utc).isoformat(), capture_year, capture_date or "",
                    file_content_hash(resolved),
                ),
            )
            image_id = int(self.connection.execute("SELECT id FROM images WHERE path = ?", (str(resolved),)).fetchone()[0])
            self.connection.execute("DELETE FROM faces WHERE image_id = ?", (image_id,))
            self.connection.executemany(
                """INSERT INTO faces(
                       image_id, face_index, identity_id, embedding, preview, intentionally_unknown
                       , bbox_x, bbox_y, bbox_width, bbox_height, unknown_group_id
                       , sharpness, profile_eligible
                       , is_art
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        image_id,
                        index,
                        identity_id,
                        embedding.astype(np.float32).tobytes(),
                        previews[index].tobytes() if previews is not None else None,
                        int(intentionally_unknown[index]),
                        *(boxes[index] if boxes[index] is not None else (None, None, None, None)),
                        unknown_group_ids[index],
                        sharpness_scores[index],
                        int(profile_eligible[index]),
                        int(is_art[index]),
                    )
                    for index, (embedding, identity_id) in enumerate(zip(embeddings, identities, strict=True))
                ],
            )
        effective_year = self.connection.execute(
            "SELECT COALESCE(capture_year_override, capture_year) FROM images WHERE id = ?",
            (image_id,),
        ).fetchone()[0]
        return CatalogImage(
            image_id, resolved, stat.st_size, stat.st_mtime_ns, len(embeddings),
            identified_count, effective_year,
        )

    def faces_for_image(self, image_id: int) -> list[CatalogFace]:
        rows = self.connection.execute(
            """SELECT faces.id, faces.identity_id, identities.name, faces.embedding, faces.preview,
                      faces.intentionally_unknown, faces.bbox_x, faces.bbox_y,
                      faces.bbox_width, faces.bbox_height, faces.unknown_group_id
                      , faces.sharpness, faces.profile_eligible
                      , faces.is_art
               FROM faces LEFT JOIN identities ON identities.id = faces.identity_id
               WHERE faces.image_id = ? ORDER BY faces.face_index""",
            (image_id,),
        ).fetchall()
        return [
            CatalogFace(
                int(row[0]), row[1], row[2], np.frombuffer(row[3], dtype=np.float32).copy(),
                np.frombuffer(row[4], dtype=np.uint8).copy() if row[4] is not None else None,
                bool(row[5]),
                (int(row[6]), int(row[7]), int(row[8]), int(row[9])) if row[6] is not None else None,
                row[10],
                row[11],
                bool(row[12]),
                bool(row[13]),
            )
            for row in rows
        ]

    def assign_face(self, face_id: int, identity_id: int, as_art: bool = False) -> None:
        with self.connection:
            image_row = self.connection.execute("SELECT image_id FROM faces WHERE id = ?", (face_id,)).fetchone()
            if not image_row:
                return
            self.connection.execute(
                """UPDATE faces SET identity_id = ?, intentionally_unknown = 0,
                   unknown_group_id = NULL, is_art = ?,
                   profile_eligible = CASE WHEN ? THEN 0 ELSE profile_eligible END
                   WHERE id = ?""",
                (identity_id, int(as_art), int(as_art), face_id),
            )
            self.connection.execute(
                """UPDATE images SET identified_count =
                   (SELECT COUNT(*) FROM faces WHERE faces.image_id = images.id AND identity_id IS NOT NULL)
                   WHERE id = ?""",
                (image_row[0],),
            )

    def mark_intentionally_unknown(self, face_id: int) -> int:
        group_id = self.create_unknown_group()
        with self.connection:
            self.connection.execute(
                """UPDATE faces SET identity_id = NULL, intentionally_unknown = 1,
                   unknown_group_id = ? WHERE id = ?""",
                (group_id, face_id),
            )
        return group_id

    def remove_face(self, face_id: int) -> None:
        """Mark a detector false-positive by removing it and refreshing image counts."""
        with self.connection:
            image_row = self.connection.execute("SELECT image_id FROM faces WHERE id = ?", (face_id,)).fetchone()
            if not image_row:
                return
            image_id = int(image_row[0])
            self.connection.execute("DELETE FROM faces WHERE id = ?", (face_id,))
            self.connection.execute(
                """UPDATE images SET
                   face_count = (SELECT COUNT(*) FROM faces WHERE faces.image_id = images.id),
                   identified_count = (SELECT COUNT(*) FROM faces WHERE faces.image_id = images.id AND identity_id IS NOT NULL)
                   WHERE id = ?""",
                (image_id,),
            )


def best_known_identity(
    embedding: np.ndarray, identities: list[KnownIdentity], threshold: float,
    capture_year: int | None = None,
) -> tuple[int | None, float]:
    best_id: int | None = None
    best_score = -1.0
    for identity in identities:
        for known in identity_embeddings_for_year(identity, capture_year):
            score = float(np.dot(embedding, known))
            if score > best_score:
                best_id, best_score = identity.identity_id, score
    return (best_id if best_score >= threshold else None), best_score


def closest_identity_matches(
    embedding: np.ndarray, identities: list[KnownIdentity], limit: int = 3,
    capture_year: int | None = None,
) -> list[tuple[KnownIdentity, float]]:
    ranked: list[tuple[KnownIdentity, float]] = []
    for identity in identities:
        if not identity.embeddings:
            continue
        score = max(float(np.dot(embedding, known)) for known in identity_embeddings_for_year(identity, capture_year))
        ranked.append((identity, score))
    ranked.sort(key=lambda item: (-item[1], item[0].name.casefold()))
    return ranked[:limit]


def best_unknown_group(
    embedding: np.ndarray, groups: list[UnknownGroup], threshold: float
) -> tuple[int | None, float]:
    best_id: int | None = None
    best_score = -1.0
    for group in groups:
        for known in group.embeddings:
            score = float(np.dot(embedding, known))
            if score > best_score:
                best_id, best_score = group.group_id, score
    return (best_id if best_score >= threshold else None), best_score


def bounded_profile(embeddings: list[np.ndarray] | tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """Keep varied samples while dropping near-duplicates and limiting drift surface."""
    selected: list[np.ndarray] = []
    for embedding in embeddings:
        if selected and max(float(np.dot(embedding, known)) for known in selected) >= PROFILE_DUPLICATE_SIMILARITY:
            continue
        selected.append(embedding)
        if len(selected) >= PROFILE_MAX_SAMPLES:
            break
    return tuple(selected)


def bounded_profile_samples(
    samples: list[tuple[np.ndarray, int | None]],
) -> tuple[tuple[np.ndarray, int | None], ...]:
    """Keep a varied profile while reserving representation for every capture year."""
    ordered = sorted(samples, key=lambda item: (item[1] is None, item[1] or 0))
    selected: list[tuple[np.ndarray, int | None]] = []
    for sample in ordered:
        if selected and max(float(np.dot(sample[0], known[0])) for known in selected) >= PROFILE_DUPLICATE_SIMILARITY:
            continue
        selected.append(sample)
    if len(selected) <= PROFILE_MAX_SAMPLES:
        return tuple(selected)
    # Round-robin years prevents a heavily photographed recent year from crowding out older appearances.
    buckets: dict[int | None, list[tuple[np.ndarray, int | None]]] = {}
    for sample in selected:
        buckets.setdefault(sample[1], []).append(sample)
    balanced: list[tuple[np.ndarray, int | None]] = []
    while len(balanced) < PROFILE_MAX_SAMPLES and any(buckets.values()):
        for year in sorted(buckets, key=lambda value: (value is None, value or 0)):
            if buckets[year] and len(balanced) < PROFILE_MAX_SAMPLES:
                balanced.append(buckets[year].pop(0))
    return tuple(balanced)


def identity_embeddings_for_year(identity: KnownIdentity, capture_year: int | None) -> tuple[np.ndarray, ...]:
    if capture_year is None or len(identity.sample_years) != len(identity.embeddings):
        return identity.embeddings
    dated = [(embedding, year) for embedding, year in zip(identity.embeddings, identity.sample_years, strict=True)
             if year is not None]
    if not dated:
        return identity.embeddings
    exact = tuple(embedding for embedding, year in dated if year == capture_year)
    if exact:
        return exact
    nearest_distance = min(abs(int(year) - capture_year) for _embedding, year in dated)
    nearest = tuple(embedding for embedding, year in dated if abs(int(year) - capture_year) == nearest_distance)
    return nearest or identity.embeddings


def image_capture_year(path: Path) -> int | None:
    """Read EXIF capture year, then a year in the filename, then filesystem modification year."""
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            for tag in (36867, 36868, 306):
                value = exif.get(tag)
                if value and (match := re.match(r"(19\d{2}|20\d{2})", str(value))):
                    return int(match.group(1))
    except (OSError, ValueError):
        pass
    if match := re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", path.stem):
        return int(match.group(1))
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).year
    except OSError:
        return None


def image_capture_date(path: Path) -> str | None:
    """Return a trustworthy ISO capture date from EXIF or a date-bearing filename."""
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            for tag in (36867, 36868, 306):
                value = exif.get(tag)
                if value and (match := re.match(
                    r"(19\d{2}|20\d{2})[:\-](\d{2})[:\-](\d{2})", str(value)
                )):
                    parsed = datetime.strptime("-".join(match.groups()), "%Y-%m-%d")
                    return parsed.date().isoformat()
    except (OSError, ValueError):
        pass
    match = re.search(
        r"(?<!\d)(19\d{2}|20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)", path.stem
    )
    if match:
        try:
            return datetime.strptime("-".join(match.groups()), "%Y-%m-%d").date().isoformat()
        except ValueError:
            pass
    return None


def file_content_hash(path: Path) -> str:
    """Return a stable SHA-256 fingerprint without loading a large photo wholly into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def default_catalog_path() -> Path:
    from .settings import default_settings_path

    return default_settings_path().with_name("face-catalog.sqlite3")
