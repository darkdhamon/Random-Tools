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
UNKNOWN_CLUSTER_SIMILARITY = 0.62


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
    fallback_embeddings: tuple[np.ndarray, ...] = ()
    fallback_sample_years: tuple[int | None, ...] = ()


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
                art_model_version INTEGER NOT NULL DEFAULT 0,
                art_kind TEXT,
                art_score REAL,
                estimated_age REAL,
                UNIQUE(image_id, face_index)
            );
            CREATE INDEX IF NOT EXISTS faces_identity_idx ON faces(identity_id);
            CREATE TABLE IF NOT EXISTS retained_identity_samples (
                source_face_id INTEGER PRIMARY KEY,
                identity_id INTEGER NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL, capture_year INTEGER,
                profile_eligible INTEGER NOT NULL DEFAULT 1,
                preview BLOB, capture_date TEXT
            );
            CREATE TABLE IF NOT EXISTS photo_identity_tags (
                image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                identity_id INTEGER NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                target_x REAL,
                target_y REAL,
                PRIMARY KEY(image_id, identity_id)
            );
            CREATE INDEX IF NOT EXISTS photo_identity_tags_identity_idx
                ON photo_identity_tags(identity_id);
            CREATE TABLE IF NOT EXISTS pet_tags (
                id INTEGER PRIMARY KEY, image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                name TEXT NOT NULL, species TEXT NOT NULL, confidence REAL,
                bbox_x REAL NOT NULL, bbox_y REAL NOT NULL,
                bbox_width REAL NOT NULL, bbox_height REAL NOT NULL,
                automatic INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
            );
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
        identity_indexes = self.connection.execute("PRAGMA index_list(identities)").fetchall()
        if any(bool(row[2]) and row[3] == "u" for row in identity_indexes):
            # Early catalogs made names unique. IDs are the durable identity key, so rebuild
            # only this parent table while foreign-key enforcement is temporarily paused.
            self.connection.commit()
            self.connection.execute("PRAGMA foreign_keys = OFF")
            try:
                with self.connection:
                    self.connection.execute(
                        """CREATE TABLE identities_without_unique_name (
                               id INTEGER PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE,
                               created_at TEXT NOT NULL, birth_year INTEGER)"""
                    )
                    self.connection.execute(
                        """INSERT INTO identities_without_unique_name(id, name, created_at, birth_year)
                           SELECT id, name, created_at, birth_year FROM identities"""
                    )
                    self.connection.execute("DROP TABLE identities")
                    self.connection.execute(
                        "ALTER TABLE identities_without_unique_name RENAME TO identities"
                    )
            finally:
                self.connection.execute("PRAGMA foreign_keys = ON")
            if self.connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise sqlite3.DatabaseError("Identity schema migration failed foreign-key validation.")
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
        tag_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(photo_identity_tags)")}
        retained_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(retained_identity_samples)")}
        if "preview" not in retained_columns:
            self.connection.execute("ALTER TABLE retained_identity_samples ADD COLUMN preview BLOB")
        if "capture_date" not in retained_columns:
            self.connection.execute("ALTER TABLE retained_identity_samples ADD COLUMN capture_date TEXT")
        if "target_x" not in tag_columns:
            self.connection.execute("ALTER TABLE photo_identity_tags ADD COLUMN target_x REAL")
        if "target_y" not in tag_columns:
            self.connection.execute("ALTER TABLE photo_identity_tags ADD COLUMN target_y REAL")
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
        if "art_model_version" not in columns:
            self.connection.execute(
                "ALTER TABLE faces ADD COLUMN art_model_version INTEGER NOT NULL DEFAULT 0"
            )
        if "art_kind" not in columns:
            self.connection.execute("ALTER TABLE faces ADD COLUMN art_kind TEXT")
        if "art_score" not in columns:
            self.connection.execute("ALTER TABLE faces ADD COLUMN art_score REAL")
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
            self.connection.execute("DELETE FROM retained_identity_samples")
            self.connection.execute("DELETE FROM images")
            self.connection.execute("DELETE FROM identities")
            self.connection.execute("DELETE FROM unknown_groups")
        self.connection.execute("VACUUM")

    def gallery_photos(
        self, search: str = "", identity_id: int | None = None, year: int | None = None,
        limit: int = 100, offset: int = 0, nsfw_filter: str = "all",
        excluded_kinds: tuple[str, ...] = (), media_kind: str | None = None,
        unknown_group_id: int | None = None,
        unknown_group_ids: tuple[int, ...] = (),
    ) -> list[dict[str, object]]:
        clauses = ["images.missing_since IS NULL"]
        values: list[object] = []
        if search:
            clauses.append("(images.path LIKE ? OR metadata.title LIKE ? OR metadata.tags LIKE ?)")
            pattern = f"%{search}%"
            values.extend((pattern, pattern, pattern))
        if identity_id is not None:
            clauses.append("""(EXISTS (SELECT 1 FROM faces WHERE faces.image_id = images.id
                                      AND faces.identity_id = ?)
                            OR EXISTS (SELECT 1 FROM photo_identity_tags tags
                                       WHERE tags.image_id = images.id AND tags.identity_id = ?))""")
            values.extend((identity_id, identity_id))
        selected_unknown_groups = tuple(dict.fromkeys(
            (*unknown_group_ids, *((unknown_group_id,) if unknown_group_id is not None else ()))
        ))
        if selected_unknown_groups:
            placeholders = ",".join("?" for _ in selected_unknown_groups)
            clauses.append(
                f"""EXISTS (SELECT 1 FROM faces WHERE faces.image_id = images.id
                             AND faces.unknown_group_id IN ({placeholders}))"""
            )
            values.extend(selected_unknown_groups)
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
                      faces.estimated_age, faces.unknown_group_id,
                      faces.bbox_x, faces.bbox_y, faces.bbox_width, faces.bbox_height,
                      faces.art_kind, faces.art_score
               FROM faces LEFT JOIN identities ON identities.id = faces.identity_id
               WHERE faces.image_id = ? ORDER BY faces.face_index""",
            (image_id,),
        ).fetchall()
        photo["faces"] = [
            {"id": int(row[0]), "name": row[1], "unknown": bool(row[2]), "art": bool(row[3]),
             "estimated_age": row[4], "unknown_group_id": row[5],
             "bbox": [int(value) for value in row[6:10]] if row[6] is not None else None,
             "art_kind": row[10], "art_score": row[11],
             "assignment_suggestions": self.face_assignment_suggestions(int(row[0]))}
            for row in faces
        ]
        photo["face_tags"] = [
            {"identity_id": int(tag[0]), "name": tag[1],
             "target_x": tag[2], "target_y": tag[3]}
            for tag in self.connection.execute(
                """SELECT identities.id, identities.name, tags.target_x, tags.target_y
                   FROM photo_identity_tags tags
                   JOIN identities ON identities.id = tags.identity_id
                   WHERE tags.image_id = ? ORDER BY identities.name COLLATE NOCASE""",
                (image_id,),
            ).fetchall()
        ]
        photo["pet_tags"] = [
            {"id": int(tag[0]), "name": tag[1], "species": tag[2], "confidence": tag[3],
             "bbox": [float(value) for value in tag[4:8]], "automatic": bool(tag[8])}
            for tag in self.connection.execute(
                """SELECT id, name, species, confidence, bbox_x, bbox_y,
                          bbox_width, bbox_height, automatic
                   FROM pet_tags WHERE image_id = ? ORDER BY id""", (image_id,)
            )
        ]
        return photo

    def pending_art_faces(self, model_version: int, limit: int = 500) -> list[tuple[int, bytes]]:
        rows = self.connection.execute(
            """SELECT id, preview FROM faces
               WHERE art_model_version < ? AND is_art = 0 AND preview IS NOT NULL
               ORDER BY id LIMIT ?""",
            (model_version, limit),
        ).fetchall()
        return [(int(row[0]), bytes(row[1])) for row in rows]

    def set_art_classification(
        self, face_id: int, kind: str, score: float, excluded: bool, model_version: int
    ) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE faces SET art_kind = ?, art_score = ?, art_model_version = ?,
                          is_art = CASE WHEN ? THEN 1 ELSE is_art END,
                          profile_eligible = CASE WHEN ? THEN 0 ELSE profile_eligible END,
                          intentionally_unknown = CASE WHEN ? THEN 0 ELSE intentionally_unknown END,
                          unknown_group_id = CASE WHEN ? THEN NULL ELSE unknown_group_id END
                   WHERE id = ?""",
                (kind, score, model_version, int(excluded), int(excluded), int(excluded),
                 int(excluded), face_id),
            )

    def face_assignment_suggestions(self, face_id: int) -> list[dict[str, object]]:
        """Rank identities by same-day presence, then biometric similarity."""
        row = self.connection.execute(
            """SELECT faces.embedding, COALESCE(images.capture_year_override, images.capture_year),
                      CASE WHEN images.capture_year_override IS NULL
                                OR CAST(substr(images.capture_date, 1, 4) AS INTEGER)
                                   = images.capture_year_override
                           THEN NULLIF(substr(images.capture_date, 1, 10), '') END
               FROM faces JOIN images ON images.id = faces.image_id WHERE faces.id = ?""",
            (face_id,),
        ).fetchone()
        if row is None:
            return []
        embedding = np.frombuffer(row[0], dtype=np.float32).copy()
        capture_year, capture_day = row[1], row[2]
        same_day_ids: set[int] = set()
        if capture_day:
            same_day_ids = {
                int(item[0]) for item in self.connection.execute(
                    """SELECT DISTINCT faces.identity_id
                       FROM faces JOIN images ON images.id = faces.image_id
                       WHERE faces.identity_id IS NOT NULL AND images.missing_since IS NULL
                             AND substr(images.capture_date, 1, 10) = ?""",
                    (capture_day,),
                )
            }
        suggestions: list[dict[str, object]] = []
        for identity in self.identities():
            samples = identity_embeddings_for_year(identity, capture_year)
            if not samples:
                samples = fallback_identity_embeddings_for_year(identity, capture_year)
            score = max((float(np.dot(embedding, known)) for known in samples), default=-1.0)
            suggestions.append({
                "id": identity.identity_id, "name": identity.name,
                "same_day": identity.identity_id in same_day_ids,
                "match_score": None if score < 0 else max(0.0, min(1.0, score)),
            })
        suggestions.sort(key=lambda item: (
            not bool(item["same_day"]),
            -(float(item["match_score"]) if item["match_score"] is not None else -1.0),
            str(item["name"]).casefold(), int(item["id"]),
        ))
        return suggestions

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
            self._retain_identity_samples([image_id])
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
            self._retain_identity_samples(unique_ids)
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
            self._retain_identity_samples(unique_ids)
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

    def set_nsfw_overrides(self, image_ids: list[int], value: int | None) -> int:
        """Apply one manual NSFW decision to multiple catalog photos."""
        if value not in (None, 0, 1):
            raise ValueError("NSFW override must be automatic, safe, or NSFW.")
        unique_ids = list(dict.fromkeys(int(image_id) for image_id in image_ids))
        if not unique_ids:
            raise ValueError("Select at least one photo.")
        placeholders = ",".join("?" for _ in unique_ids)
        found = int(self.connection.execute(
            f"SELECT COUNT(*) FROM images WHERE id IN ({placeholders})", unique_ids
        ).fetchone()[0])
        if found != len(unique_ids):
            raise ValueError("One or more selected photos were not found in the catalog.")
        with self.connection:
            self.connection.executemany(
                """INSERT INTO image_metadata(image_id, nsfw_override) VALUES (?, ?)
                   ON CONFLICT(image_id) DO UPDATE SET nsfw_override=excluded.nsfw_override""",
                [(image_id, value) for image_id in unique_ids],
            )
        return len(unique_ids)

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
                      faces.embedding, COALESCE(images.capture_year_override, images.capture_year),
                      faces.profile_eligible
               FROM identities LEFT JOIN faces ON faces.identity_id = identities.id
                    AND faces.is_art = 0
               LEFT JOIN images ON images.id = faces.image_id
               ORDER BY identities.name, faces.id"""
        ).fetchall()
        rows.extend(self.connection.execute(
            """SELECT identities.id, identities.name, identities.birth_year,
                      retained.embedding, retained.capture_year, retained.profile_eligible
               FROM retained_identity_samples retained
               JOIN identities ON identities.id = retained.identity_id
               ORDER BY identities.name, retained.source_face_id"""
        ).fetchall())
        grouped: dict[tuple[int, str, int | None], dict[bool, list[tuple[np.ndarray, int | None]]]] = {}
        for identity_id, name, birth_year, blob, capture_year, profile_eligible in rows:
            grouped.setdefault((identity_id, name, birth_year), {True: [], False: []})
            if blob is not None:
                grouped[(identity_id, name, birth_year)][bool(profile_eligible)].append(
                    (np.frombuffer(blob, dtype=np.float32).copy(), capture_year)
                )
        identities: list[KnownIdentity] = []
        for key, values in grouped.items():
            samples = bounded_profile_samples(values[True])
            fallback = bounded_profile_samples(values[False])
            identities.append(KnownIdentity(key[0], key[1], tuple(v[0] for v in samples),
                                            tuple(v[1] for v in samples), key[2],
                                            tuple(v[0] for v in fallback),
                                            tuple(v[1] for v in fallback)))
        return identities

    def gallery_identity_ids(self) -> set[int]:
        """Identity IDs that still occur in at least one available library photo."""
        return {
            int(row[0]) for row in self.connection.execute(
                """SELECT faces.identity_id FROM faces JOIN images ON images.id = faces.image_id
                   WHERE faces.identity_id IS NOT NULL AND images.missing_since IS NULL
                   UNION
                   SELECT tags.identity_id FROM photo_identity_tags tags
                   JOIN images ON images.id = tags.image_id WHERE images.missing_since IS NULL"""
            )
        }

    def _retain_identity_samples(self, image_ids: list[int]) -> None:
        if not image_ids:
            return
        placeholders = ",".join("?" for _ in image_ids)
        self.connection.execute(
            f"""INSERT OR IGNORE INTO retained_identity_samples(
                       source_face_id, identity_id, embedding, capture_year, profile_eligible,
                       preview, capture_date)
                SELECT faces.id, faces.identity_id, faces.embedding,
                       COALESCE(images.capture_year_override, images.capture_year),
                       faces.profile_eligible, faces.preview,
                       CASE WHEN images.capture_year_override IS NULL
                                  OR CAST(substr(images.capture_date, 1, 4) AS INTEGER)
                                     = images.capture_year_override
                            THEN NULLIF(images.capture_date, '') END
                FROM faces JOIN images ON images.id = faces.image_id
                WHERE faces.image_id IN ({placeholders}) AND faces.identity_id IS NOT NULL
                      AND faces.is_art = 0 AND faces.embedding IS NOT NULL""",
            image_ids,
        )

    def set_identity_birth_year(self, identity_id: int, birth_year: int | None) -> None:
        if birth_year is not None and not 1900 <= birth_year <= datetime.now().year:
            raise ValueError("Birth year must be between 1900 and the current year.")
        with self.connection:
            self.connection.execute(
                "UPDATE identities SET birth_year = ? WHERE id = ?", (birth_year, identity_id)
            )

    def identity_summaries(self) -> list[dict[str, object]]:
        """Return lightweight identity-management rows without loading biometric vectors."""
        rows = self.connection.execute(
            """SELECT identities.id, identities.name, identities.birth_year,
                      COUNT(CASE WHEN images.missing_since IS NULL THEN faces.id END),
                      COUNT(DISTINCT CASE WHEN images.missing_since IS NULL THEN faces.image_id END),
                      COALESCE(SUM(CASE WHEN faces.profile_eligible = 1 THEN 1 ELSE 0 END), 0)
                      + (SELECT COUNT(*) FROM retained_identity_samples retained
                         WHERE retained.identity_id = identities.id
                               AND retained.profile_eligible = 1),
                      MAX(CASE WHEN images.missing_since IS NULL THEN
                            CASE WHEN images.capture_year_override IS NULL
                                       OR CAST(substr(images.capture_date, 1, 4) AS INTEGER)
                                          = images.capture_year_override
                                 THEN COALESCE(NULLIF(images.capture_date, ''),
                                               CAST(images.capture_year AS TEXT))
                                 ELSE CAST(images.capture_year_override AS TEXT) END
                          END)
               FROM identities
               LEFT JOIN faces ON faces.identity_id = identities.id
               LEFT JOIN images ON images.id = faces.image_id
               GROUP BY identities.id, identities.name, identities.birth_year
               ORDER BY identities.name COLLATE NOCASE"""
        ).fetchall()
        reference_rows = self.connection.execute(
            """SELECT faces.identity_id, faces.id, faces.embedding,
                      COALESCE(images.capture_year_override, images.capture_year),
                      CASE WHEN images.capture_year_override IS NULL
                                  OR CAST(substr(images.capture_date, 1, 4) AS INTEGER)
                                     = images.capture_year_override
                           THEN NULLIF(images.capture_date, '') END
               FROM faces JOIN images ON images.id = faces.image_id
               WHERE faces.identity_id IS NOT NULL AND faces.profile_eligible = 1
                     AND faces.preview IS NOT NULL
               ORDER BY faces.identity_id, faces.id"""
        ).fetchall()
        reference_rows.extend(self.connection.execute(
            """SELECT identity_id, -source_face_id, embedding, capture_year, capture_date
               FROM retained_identity_samples
               WHERE profile_eligible = 1 AND preview IS NOT NULL
               ORDER BY identity_id, source_face_id"""
        ).fetchall())
        references: dict[int, list[tuple[int, np.ndarray, int | None, str | None]]] = {}
        for identity_id, face_id, embedding, capture_year, capture_date in reference_rows:
            references.setdefault(int(identity_id), []).append(
                (int(face_id), np.frombuffer(embedding, dtype=np.float32).copy(),
                 capture_year, capture_date)
            )

        def selected_face_ids(identity_id: int) -> list[int]:
            records = sorted(
                references.get(identity_id, []), key=lambda item: (item[2] is None, item[2] or 0)
            )
            selected: list[tuple[int, np.ndarray, int | None, str | None]] = []
            for record in records:
                if selected and max(
                    float(np.dot(record[1], known[1])) for known in selected
                ) >= PROFILE_DUPLICATE_SIMILARITY:
                    continue
                selected.append(record)
            if len(selected) > PROFILE_MAX_SAMPLES:
                buckets: dict[int | None, list[tuple[int, np.ndarray, int | None, str | None]]] = {}
                for record in selected:
                    buckets.setdefault(record[2], []).append(record)
                balanced: list[tuple[int, np.ndarray, int | None, str | None]] = []
                while len(balanced) < PROFILE_MAX_SAMPLES and any(buckets.values()):
                    for year in sorted(buckets, key=lambda value: (value is None, value or 0)):
                        if buckets[year] and len(balanced) < PROFILE_MAX_SAMPLES:
                            balanced.append(buckets[year].pop(0))
                selected = balanced
            selected.sort(
                key=lambda record: (
                    record[3] or (f"{record[2]:04d}" if record[2] is not None else ""),
                    record[2] or 0,
                    record[0],
                ),
                reverse=True,
            )
            return [record[0] for record in selected]

        current_year = datetime.now().year
        summaries = []
        for row in rows:
            identity_id = int(row[0])
            summaries.append({
                "id": identity_id, "name": row[1], "birth_year": row[2],
                "age": current_year - int(row[2]) if row[2] is not None else None,
                "face_count": int(row[3]), "photo_count": int(row[4]),
                "profile_sample_count": int(row[5]),
                "last_seen": row[6],
                "reference_face_ids": selected_face_ids(identity_id),
            })
        return summaries

    def identity_reference_preview(self, face_id: int) -> bytes | None:
        if face_id < 0:
            row = self.connection.execute(
                """SELECT preview FROM retained_identity_samples
                   WHERE source_face_id = ? AND profile_eligible = 1 AND preview IS NOT NULL""",
                (-face_id,),
            ).fetchone()
            return bytes(row[0]) if row else None
        row = self.connection.execute(
            """SELECT faces.preview FROM faces JOIN images ON images.id = faces.image_id
               WHERE faces.id = ? AND faces.identity_id IS NOT NULL
                     AND faces.profile_eligible = 1 AND faces.preview IS NOT NULL""",
            (face_id,),
        ).fetchone()
        return bytes(row[0]) if row else None

    def unidentified_summaries(self) -> list[dict[str, object]]:
        """Cluster similar anonymous groups for identity-management review."""
        rows = self.connection.execute(
            """SELECT unknown_groups.id,
                      COUNT(CASE WHEN images.missing_since IS NULL THEN faces.id END),
                      COUNT(DISTINCT CASE WHEN images.missing_since IS NULL THEN faces.image_id END),
                      MAX(CASE WHEN images.missing_since IS NULL THEN
                            CASE WHEN images.capture_year_override IS NULL
                                       OR CAST(substr(images.capture_date, 1, 4) AS INTEGER)
                                          = images.capture_year_override
                                 THEN COALESCE(NULLIF(images.capture_date, ''),
                                               CAST(images.capture_year AS TEXT))
                                 ELSE CAST(images.capture_year_override AS TEXT) END
                          END)
               FROM unknown_groups
               LEFT JOIN faces ON faces.unknown_group_id = unknown_groups.id
               LEFT JOIN images ON images.id = faces.image_id
               GROUP BY unknown_groups.id
               HAVING COUNT(CASE WHEN images.missing_since IS NULL THEN faces.id END) > 0
               ORDER BY unknown_groups.id DESC"""
        ).fetchall()
        group_ids = [int(row[0]) for row in rows]
        profiles = {group.group_id: group.embeddings for group in self.unknown_groups()}
        parent = {group_id: group_id for group_id in group_ids}

        def find(group_id: int) -> int:
            while parent[group_id] != group_id:
                parent[group_id] = parent[parent[group_id]]
                group_id = parent[group_id]
            return group_id

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        for index, first in enumerate(group_ids):
            first_profile = profiles.get(first, ())
            if not first_profile:
                continue
            for second in group_ids[index + 1:]:
                second_profile = profiles.get(second, ())
                if second_profile and max(
                    float(np.dot(left, right))
                    for left in first_profile for right in second_profile
                ) >= UNKNOWN_CLUSTER_SIMILARITY:
                    union(first, second)

        clusters: dict[int, list[int]] = {}
        for group_id in group_ids:
            clusters.setdefault(find(group_id), []).append(group_id)
        counts = {int(row[0]): (int(row[1]), int(row[2])) for row in rows}
        last_seen = {int(row[0]): row[3] for row in rows}
        previews: dict[int, list[int]] = {}
        photo_ids: dict[int, set[int]] = {}
        for group_id, face_id, image_id, has_preview in self.connection.execute(
            """SELECT faces.unknown_group_id, faces.id, faces.image_id,
                      faces.preview IS NOT NULL
               FROM faces JOIN images ON images.id = faces.image_id
               WHERE faces.unknown_group_id IS NOT NULL AND images.missing_since IS NULL
               ORDER BY faces.unknown_group_id, faces.profile_eligible DESC, faces.id DESC"""
        ).fetchall():
            group_id = int(group_id)
            photo_ids.setdefault(group_id, set()).add(int(image_id))
            group_previews = previews.setdefault(group_id, [])
            if len(group_previews) < 12 and has_preview:
                group_previews.append(int(face_id))
        summaries = []
        for members in clusters.values():
            members.sort(reverse=True)
            cluster_previews = [face_id for group_id in members for face_id in previews.get(group_id, [])][:12]
            distinct_photos = set().union(*(photo_ids.get(group_id, set()) for group_id in members))
            summaries.append({
                "id": members[0], "group_ids": members,
                "label": (f"Unidentified person {members[0]}" if len(members) == 1
                          else f"Similar unidentified people ({len(members)} groups)"),
                "face_count": sum(counts[group_id][0] for group_id in members),
                "photo_count": len(distinct_photos), "reference_face_ids": cluster_previews,
                "last_seen": max((last_seen[group_id] or "" for group_id in members), default="") or None,
            })
        summaries.sort(key=lambda item: int(item["id"]), reverse=True)
        return summaries

    def unidentified_reference_preview(self, face_id: int) -> bytes | None:
        row = self.connection.execute(
            """SELECT faces.preview FROM faces JOIN images ON images.id = faces.image_id
               WHERE faces.id = ? AND faces.unknown_group_id IS NOT NULL
                     AND faces.preview IS NOT NULL AND images.missing_since IS NULL""",
            (face_id,),
        ).fetchone()
        return bytes(row[0]) if row else None

    def update_identity(self, identity_id: int, name: str, birth_year: int | None) -> None:
        """Rename an identity and update its birth year while preserving its stable ID."""
        clean_name = " ".join(name.split())
        if not clean_name:
            raise ValueError("An identity name is required.")
        if birth_year is not None and not 1900 <= birth_year <= datetime.now().year:
            raise ValueError("Birth year must be between 1900 and the current year.")
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE identities SET name = ?, birth_year = ? WHERE id = ?",
                (clean_name, birth_year, identity_id),
            )
        if cursor.rowcount != 1:
            raise ValueError("Identity was not found.")

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

    def assign_unknown_groups(self, group_ids: list[int], identity_id: int) -> tuple[int, int]:
        """Attach a review cluster of anonymous groups to one known identity."""
        selected = list(dict.fromkeys(int(value) for value in group_ids))
        if not selected:
            raise ValueError("Select at least one unidentified group.")
        if self.connection.execute("SELECT 1 FROM identities WHERE id = ?", (identity_id,)).fetchone() is None:
            raise ValueError("The selected known identity no longer exists.")
        placeholders = ",".join("?" for _ in selected)
        existing = self.connection.execute(
            f"SELECT id FROM unknown_groups WHERE id IN ({placeholders})", selected
        ).fetchall()
        if len(existing) != len(selected):
            raise ValueError("One or more unidentified groups no longer exist.")
        image_ids = [int(row[0]) for row in self.connection.execute(
            f"SELECT DISTINCT image_id FROM faces WHERE unknown_group_id IN ({placeholders})", selected
        ).fetchall()]
        with self.connection:
            cursor = self.connection.execute(
                f"""UPDATE faces SET identity_id = ?, intentionally_unknown = 0,
                           unknown_group_id = NULL
                    WHERE unknown_group_id IN ({placeholders})""",
                (identity_id, *selected),
            )
            for image_id in image_ids:
                self.connection.execute(
                    """UPDATE images SET identified_count =
                       (SELECT COUNT(*) FROM faces WHERE faces.image_id = images.id
                        AND identity_id IS NOT NULL) WHERE id = ?""", (image_id,),
                )
            self.connection.execute(
                f"DELETE FROM unknown_groups WHERE id IN ({placeholders})", selected
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

    def create_identity(self, name: str, birth_year: int | None = None) -> int:
        """Create a distinct identity even when another profile uses the same display name."""
        clean_name = " ".join(name.split())
        if not clean_name:
            raise ValueError("An identity name is required.")
        if birth_year is not None and not 1900 <= birth_year <= datetime.now().year:
            raise ValueError("Birth year must be between 1900 and the current year.")
        cursor = self.connection.execute(
            "INSERT INTO identities(name, created_at, birth_year) VALUES (?, ?, ?)",
            (clean_name, datetime.now(timezone.utc).isoformat(), birth_year),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def merge_identities(self, identity_ids: list[int], target_identity_id: int) -> tuple[int, int]:
        """Merge selected profiles into one stable target identity."""
        selected = list(dict.fromkeys(int(value) for value in identity_ids))
        if target_identity_id not in selected or len(selected) < 2:
            raise ValueError("Select at least two identities and choose one selected identity to keep.")
        placeholders = ",".join("?" for _ in selected)
        existing = self.connection.execute(
            f"SELECT id FROM identities WHERE id IN ({placeholders})", selected
        ).fetchall()
        if len(existing) != len(selected):
            raise ValueError("One or more selected identities no longer exist.")
        sources = [value for value in selected if value != target_identity_id]
        source_placeholders = ",".join("?" for _ in sources)
        with self.connection:
            face_cursor = self.connection.execute(
                f"UPDATE faces SET identity_id = ? WHERE identity_id IN ({source_placeholders})",
                (target_identity_id, *sources),
            )
            self.connection.execute(
                f"UPDATE retained_identity_samples SET identity_id = ? WHERE identity_id IN ({source_placeholders})",
                (target_identity_id, *sources),
            )
            self.connection.execute(
                f"""INSERT OR IGNORE INTO photo_identity_tags(
                           image_id, identity_id, created_at, target_x, target_y)
                    SELECT image_id, ?, created_at, target_x, target_y FROM photo_identity_tags
                    WHERE identity_id IN ({source_placeholders})""",
                (target_identity_id, *sources),
            )
            tag_cursor = self.connection.execute(
                f"DELETE FROM photo_identity_tags WHERE identity_id IN ({source_placeholders})",
                sources,
            )
            self.connection.execute(
                f"DELETE FROM identities WHERE id IN ({source_placeholders})", sources
            )
        return int(face_cursor.rowcount), int(tag_cursor.rowcount)

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
            self._retain_identity_samples(image_ids)
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
                   AND NOT EXISTS (SELECT 1 FROM faces WHERE identity_id = identities.id)
                   AND NOT EXISTS (SELECT 1 FROM photo_identity_tags
                                   WHERE identity_id = identities.id)""",
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
        art_kinds: list[str | None] | None = None,
        art_scores: list[float | None] | None = None,
        art_model_version: int = 0,
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
        if art_kinds is None:
            art_kinds = [None] * len(embeddings)
        if art_scores is None:
            art_scores = [None] * len(embeddings)
        if len(art_kinds) != len(embeddings) or len(art_scores) != len(embeddings):
            raise ValueError("Every face must have artwork-classification values.")
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
                       , is_art, art_kind, art_score, art_model_version
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                        art_kinds[index],
                        art_scores[index],
                        art_model_version,
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

    def add_photo_identity_tag(
        self, image_id: int, identity_id: int,
        target_x: float | None = None, target_y: float | None = None,
    ) -> None:
        """Tag a person in a photo without treating the tag as biometric evidence."""
        if (target_x is None) != (target_y is None):
            raise ValueError("Both target coordinates are required.")
        if target_x is not None and not (0.0 <= target_x <= 1.0 and 0.0 <= target_y <= 1.0):
            raise ValueError("Target coordinates must be within the image.")
        with self.connection:
            cursor = self.connection.execute(
                """INSERT INTO photo_identity_tags(
                         image_id, identity_id, created_at, target_x, target_y)
                   SELECT ?, ?, ?, ?, ?
                   WHERE EXISTS (SELECT 1 FROM images WHERE id = ? AND missing_since IS NULL)
                     AND EXISTS (SELECT 1 FROM identities WHERE id = ?)
                   ON CONFLICT(image_id, identity_id) DO UPDATE SET
                       target_x = COALESCE(excluded.target_x, target_x),
                       target_y = COALESCE(excluded.target_y, target_y)""",
                (image_id, identity_id, datetime.now(timezone.utc).isoformat(),
                 target_x, target_y, image_id, identity_id),
            )
        if cursor.rowcount == 0:
            existing = self.connection.execute(
                "SELECT 1 FROM photo_identity_tags WHERE image_id = ? AND identity_id = ?",
                (image_id, identity_id),
            ).fetchone()
            if existing is None:
                raise ValueError("Photo or identity was not found.")

    def remove_photo_identity_tag(self, image_id: int, identity_id: int) -> None:
        """Remove a non-biometric person tag from a photo."""
        with self.connection:
            self.connection.execute(
                "DELETE FROM photo_identity_tags WHERE image_id = ? AND identity_id = ?",
                (image_id, identity_id),
            )

    def add_pet_tag(
        self, image_id: int, name: str, species: str,
        bbox: tuple[float, float, float, float], confidence: float | None = None,
        automatic: bool = False,
    ) -> int:
        clean_name, clean_species = " ".join(name.split()), " ".join(species.split())
        if not clean_name or not clean_species:
            raise ValueError("Pet name and species are required.")
        x, y, width, height = bbox
        if not all(0.0 <= value <= 1.0 for value in bbox) or x + width > 1.001 or y + height > 1.001:
            raise ValueError("Pet target must be within the image.")
        if self.connection.execute(
            "SELECT 1 FROM images WHERE id = ? AND missing_since IS NULL", (image_id,)
        ).fetchone() is None:
            raise ValueError("Photo was not found.")
        with self.connection:
            cursor = self.connection.execute(
                """INSERT INTO pet_tags(image_id, name, species, confidence, bbox_x, bbox_y,
                           bbox_width, bbox_height, automatic, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (image_id, clean_name, clean_species, confidence, x, y, width, height,
                 int(automatic), datetime.now(timezone.utc).isoformat()),
            )
        return int(cursor.lastrowid)

    def update_pet_tag(
        self, tag_id: int, name: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        clean_name = " ".join(name.split()) if name is not None else None
        if name is not None and not clean_name:
            raise ValueError("Pet name is required.")
        if bbox is not None:
            x, y, width, height = bbox
            if not all(0.0 <= value <= 1.0 for value in bbox) or x + width > 1.001 or y + height > 1.001:
                raise ValueError("Pet target must be within the image.")
        with self.connection:
            cursor = self.connection.execute(
                """UPDATE pet_tags SET name = COALESCE(?, name),
                          bbox_x = COALESCE(?, bbox_x), bbox_y = COALESCE(?, bbox_y),
                          bbox_width = COALESCE(?, bbox_width), bbox_height = COALESCE(?, bbox_height)
                   WHERE id = ?""",
                (clean_name, *(bbox or (None, None, None, None)), tag_id),
            )
        if cursor.rowcount == 0:
            raise ValueError("Pet tag was not found.")

    def remove_pet_tag(self, tag_id: int) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM pet_tags WHERE id = ?", (tag_id,))

    def clear_automatic_pet_tags(self, image_id: int) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM pet_tags WHERE image_id = ? AND automatic = 1", (image_id,)
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
        if face_id < 0:
            with self.connection:
                self.connection.execute(
                    "DELETE FROM retained_identity_samples WHERE source_face_id = ?",
                    (-face_id,),
                )
            return
        with self.connection:
            image_row = self.connection.execute(
                "SELECT image_id, unknown_group_id FROM faces WHERE id = ?", (face_id,)
            ).fetchone()
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
            if image_row[1] is not None:
                self.connection.execute(
                    """DELETE FROM unknown_groups WHERE id = ?
                       AND NOT EXISTS (SELECT 1 FROM faces WHERE unknown_group_id = ?)""",
                    (image_row[1], image_row[1]),
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
    if best_score >= threshold:
        return best_id, best_score
    # Low-quality assigned samples are deliberately isolated from the primary profile.
    # They are considered only when no clear sample can produce an acceptable match.
    fallback_id: int | None = None
    fallback_score = -1.0
    for identity in identities:
        for known in fallback_identity_embeddings_for_year(identity, capture_year):
            score = float(np.dot(embedding, known))
            if score > fallback_score:
                fallback_id, fallback_score = identity.identity_id, score
    if fallback_score > best_score:
        best_id, best_score = fallback_id, fallback_score
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


def fallback_identity_embeddings_for_year(
    identity: KnownIdentity, capture_year: int | None
) -> tuple[np.ndarray, ...]:
    fallback = KnownIdentity(
        identity.identity_id, identity.name, identity.fallback_embeddings,
        identity.fallback_sample_years, identity.birth_year,
    )
    return identity_embeddings_for_year(fallback, capture_year)


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
