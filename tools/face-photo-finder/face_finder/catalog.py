from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import numpy as np


@dataclass(frozen=True)
class KnownIdentity:
    identity_id: int
    name: str
    embeddings: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class CatalogFace:
    face_id: int
    identity_id: int | None
    identity_name: str | None
    embedding: np.ndarray
    preview: np.ndarray | None
    intentionally_unknown: bool


@dataclass(frozen=True)
class CatalogImage:
    image_id: int
    path: Path
    size: int
    modified_ns: int
    face_count: int
    identified_count: int


class FaceCatalog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS identities (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                size INTEGER NOT NULL,
                modified_ns INTEGER NOT NULL,
                face_count INTEGER NOT NULL,
                identified_count INTEGER NOT NULL,
                scanned_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY,
                image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                face_index INTEGER NOT NULL,
                identity_id INTEGER REFERENCES identities(id) ON DELETE SET NULL,
                embedding BLOB NOT NULL,
                preview BLOB,
                intentionally_unknown INTEGER NOT NULL DEFAULT 0,
                UNIQUE(image_id, face_index)
            );
            CREATE INDEX IF NOT EXISTS faces_identity_idx ON faces(identity_id);
            """
        )
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(faces)")}
        if "preview" not in columns:
            self.connection.execute("ALTER TABLE faces ADD COLUMN preview BLOB")
        if "intentionally_unknown" not in columns:
            self.connection.execute(
                "ALTER TABLE faces ADD COLUMN intentionally_unknown INTEGER NOT NULL DEFAULT 0"
            )

    def close(self) -> None:
        self.connection.close()

    def identities(self) -> list[KnownIdentity]:
        rows = self.connection.execute(
            """SELECT identities.id, identities.name, faces.embedding
               FROM identities LEFT JOIN faces ON faces.identity_id = identities.id
               ORDER BY identities.name, faces.id"""
        ).fetchall()
        grouped: dict[tuple[int, str], list[np.ndarray]] = {}
        for identity_id, name, blob in rows:
            grouped.setdefault((identity_id, name), [])
            if blob is not None:
                grouped[(identity_id, name)].append(np.frombuffer(blob, dtype=np.float32).copy())
        return [KnownIdentity(key[0], key[1], tuple(values)) for key, values in grouped.items()]

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

    def cached_image(self, path: Path) -> CatalogImage | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        row = self.connection.execute(
            "SELECT id, path, size, modified_ns, face_count, identified_count FROM images WHERE path = ?",
            (str(path.resolve()),),
        ).fetchone()
        if not row or row[2] != stat.st_size or row[3] != stat.st_mtime_ns:
            return None
        return CatalogImage(int(row[0]), Path(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]))

    def store_scan(
        self,
        path: Path,
        embeddings: list[np.ndarray],
        identities: list[int | None],
        previews: list[np.ndarray] | None = None,
        intentionally_unknown: list[bool] | None = None,
    ) -> CatalogImage:
        if len(embeddings) != len(identities):
            raise ValueError("Every face must have an identity assignment.")
        if previews is not None and len(previews) != len(embeddings):
            raise ValueError("Every face must have a corresponding preview.")
        if intentionally_unknown is None:
            intentionally_unknown = [False] * len(embeddings)
        if len(intentionally_unknown) != len(embeddings):
            raise ValueError("Every face must have an unknown-person status.")
        resolved = path.resolve()
        stat = resolved.stat()
        identified_count = sum(value is not None for value in identities)
        with self.connection:
            self.connection.execute(
                """INSERT INTO images(path, size, modified_ns, face_count, identified_count, scanned_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET size=excluded.size, modified_ns=excluded.modified_ns,
                   face_count=excluded.face_count, identified_count=excluded.identified_count,
                   scanned_at=excluded.scanned_at""",
                (
                    str(resolved), stat.st_size, stat.st_mtime_ns, len(embeddings), identified_count,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            image_id = int(self.connection.execute("SELECT id FROM images WHERE path = ?", (str(resolved),)).fetchone()[0])
            self.connection.execute("DELETE FROM faces WHERE image_id = ?", (image_id,))
            self.connection.executemany(
                """INSERT INTO faces(
                       image_id, face_index, identity_id, embedding, preview, intentionally_unknown
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (
                        image_id,
                        index,
                        identity_id,
                        embedding.astype(np.float32).tobytes(),
                        previews[index].tobytes() if previews is not None else None,
                        int(intentionally_unknown[index]),
                    )
                    for index, (embedding, identity_id) in enumerate(zip(embeddings, identities, strict=True))
                ],
            )
        return CatalogImage(image_id, resolved, stat.st_size, stat.st_mtime_ns, len(embeddings), identified_count)

    def faces_for_image(self, image_id: int) -> list[CatalogFace]:
        rows = self.connection.execute(
            """SELECT faces.id, faces.identity_id, identities.name, faces.embedding, faces.preview,
                      faces.intentionally_unknown
               FROM faces LEFT JOIN identities ON identities.id = faces.identity_id
               WHERE faces.image_id = ? ORDER BY faces.face_index""",
            (image_id,),
        ).fetchall()
        return [
            CatalogFace(
                int(row[0]), row[1], row[2], np.frombuffer(row[3], dtype=np.float32).copy(),
                np.frombuffer(row[4], dtype=np.uint8).copy() if row[4] is not None else None,
                bool(row[5]),
            )
            for row in rows
        ]

    def assign_face(self, face_id: int, identity_id: int) -> None:
        with self.connection:
            image_row = self.connection.execute("SELECT image_id FROM faces WHERE id = ?", (face_id,)).fetchone()
            if not image_row:
                return
            self.connection.execute(
                "UPDATE faces SET identity_id = ?, intentionally_unknown = 0 WHERE id = ?",
                (identity_id, face_id),
            )
            self.connection.execute(
                """UPDATE images SET identified_count =
                   (SELECT COUNT(*) FROM faces WHERE faces.image_id = images.id AND identity_id IS NOT NULL)
                   WHERE id = ?""",
                (image_row[0],),
            )

    def mark_intentionally_unknown(self, face_id: int) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE faces SET identity_id = NULL, intentionally_unknown = 1 WHERE id = ?", (face_id,)
            )

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
    embedding: np.ndarray, identities: list[KnownIdentity], threshold: float
) -> tuple[int | None, float]:
    best_id: int | None = None
    best_score = -1.0
    for identity in identities:
        for known in identity.embeddings:
            score = float(np.dot(embedding, known))
            if score > best_score:
                best_id, best_score = identity.identity_id, score
    return (best_id if best_score >= threshold else None), best_score


def default_catalog_path() -> Path:
    from .settings import default_settings_path

    return default_settings_path().with_name("face-catalog.sqlite3")
