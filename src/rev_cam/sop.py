from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .engineering import WORK_CENTRES

_DB_PATH = Path(__file__).resolve().parent / "engineering.sqlite3"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class StandardOperatingProcedure:
    id: int
    title: str
    reference: str
    work_centre_id: int
    instructions: str
    revision: int
    is_active: bool
    created_at: str
    updated_at: str

    def serialise(self, *, include_work_centre: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "title": self.title,
            "reference": self.reference,
            "work_centre": self.work_centre_id,
            "instructions": self.instructions,
            "revision": self.revision,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_work_centre:
            work_centre = WORK_CENTRES.get(self.work_centre_id)
            payload["work_centre_name"] = work_centre.name if work_centre else None
        return payload


class SopStore:
    def __init__(self, path: Path = _DB_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialise(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS standard_operating_procedures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    reference TEXT NOT NULL UNIQUE,
                    work_centre_id INTEGER NOT NULL REFERENCES work_centres(id) ON DELETE RESTRICT,
                    instructions TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def list(self, *, work_centre_id: int | None = None) -> list[StandardOperatingProcedure]:
        query = "SELECT * FROM standard_operating_procedures"
        params: tuple[object, ...]
        if work_centre_id is not None:
            query += " WHERE work_centre_id = ?"
            params = (work_centre_id,)
        else:
            params = ()
        query += " ORDER BY title"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_model(row) for row in rows]

    def create(
        self,
        *,
        title: str,
        reference: str,
        work_centre_id: int,
        instructions: str,
        revision: int = 1,
        is_active: bool = True,
    ) -> StandardOperatingProcedure:
        title = title.strip()
        reference = reference.strip()
        instructions = instructions.strip()
        if not title:
            raise ValueError("Title is required.")
        if not reference:
            raise ValueError("Reference is required.")
        if WORK_CENTRES.get(work_centre_id) is None:
            raise ValueError("Work centre does not exist.")
        timestamp = _utcnow()
        with self._lock, self._connect() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO standard_operating_procedures (
                        title, reference, work_centre_id, instructions, revision, is_active, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        title,
                        reference,
                        work_centre_id,
                        instructions,
                        int(revision),
                        1 if is_active else 0,
                        timestamp,
                        timestamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise _map_sop_integrity_error(exc) from exc
            sop_id = cursor.lastrowid
        result = self.get(sop_id)
        if result is None:  # pragma: no cover - sanity guard
            raise RuntimeError("Unable to load SOP after creation.")
        return result

    def update(self, sop_id: int, **fields: object) -> StandardOperatingProcedure:
        allowed = {"title", "reference", "work_centre_id", "instructions", "revision", "is_active"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            raise ValueError("No valid fields provided for update.")
        if "title" in updates:
            title = str(updates["title"]).strip()
            if not title:
                raise ValueError("Title is required.")
            updates["title"] = title
        if "reference" in updates:
            reference = str(updates["reference"]).strip()
            if not reference:
                raise ValueError("Reference is required.")
            updates["reference"] = reference
        if "work_centre_id" in updates:
            work_centre_id = int(updates["work_centre_id"])
            if WORK_CENTRES.get(work_centre_id) is None:
                raise ValueError("Work centre does not exist.")
            updates["work_centre_id"] = work_centre_id
        if "instructions" in updates:
            updates["instructions"] = str(updates["instructions"]).strip()
        if "revision" in updates:
            updates["revision"] = int(updates["revision"])
        if "is_active" in updates:
            updates["is_active"] = 1 if bool(updates["is_active"]) else 0
        assignments = ", ".join(f"{field} = ?" for field in updates)
        params = [updates[field] for field in updates]
        params.append(_utcnow())
        params.append(sop_id)
        with self._lock, self._connect() as conn:
            try:
                updated = conn.execute(
                    f"UPDATE standard_operating_procedures SET {assignments}, updated_at = ? WHERE id = ?",
                    params,
                )
            except sqlite3.IntegrityError as exc:
                raise _map_sop_integrity_error(exc) from exc
            if updated.rowcount == 0:
                raise KeyError(sop_id)
        result = self.get(sop_id)
        if result is None:  # pragma: no cover - sanity guard
            raise KeyError(sop_id)
        return result

    def get(self, sop_id: int) -> StandardOperatingProcedure | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM standard_operating_procedures WHERE id = ?",
                (sop_id,),
            ).fetchone()
        return self._row_to_model(row) if row else None

    @staticmethod
    def _row_to_model(row: sqlite3.Row | None) -> StandardOperatingProcedure:
        assert row is not None
        return StandardOperatingProcedure(
            id=int(row["id"]),
            title=str(row["title"]),
            reference=str(row["reference"]),
            work_centre_id=int(row["work_centre_id"]),
            instructions=str(row["instructions"]),
            revision=int(row["revision"]),
            is_active=bool(row["is_active"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )


def _map_sop_integrity_error(exc: sqlite3.IntegrityError) -> ValueError:
    message = str(exc).lower()
    if "reference" in message:
        return ValueError("A SOP with this reference already exists.")
    return ValueError("Unable to persist SOP.")


SOPS = SopStore()


def reset_sops() -> None:
    SOPS._initialise()  # type: ignore[attr-defined]
