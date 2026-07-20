from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3

import numpy as np
import cv2
from PIL import Image

PROFILE_MAX_SAMPLES = 64
PROFILE_DUPLICATE_SIMILARITY = 0.92


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
                capture_year_override INTEGER
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
                      images.capture_year_override IS NOT NULL
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
            )
            for row in rows
        ]

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
        identified_count = sum(value is not None for value in identities)
        with self.connection:
            self.connection.execute(
                """INSERT INTO images(path, size, modified_ns, face_count, identified_count, scanned_at, capture_year)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET size=excluded.size, modified_ns=excluded.modified_ns,
                   face_count=excluded.face_count, identified_count=excluded.identified_count,
                   scanned_at=excluded.scanned_at, capture_year=excluded.capture_year""",
                (
                    str(resolved), stat.st_size, stat.st_mtime_ns, len(embeddings), identified_count,
                    datetime.now(timezone.utc).isoformat(), capture_year,
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


def default_catalog_path() -> Path:
    from .settings import default_settings_path

    return default_settings_path().with_name("face-catalog.sqlite3")
