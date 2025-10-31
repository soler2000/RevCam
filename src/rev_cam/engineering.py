from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

_DB_PATH = Path(__file__).resolve().parent / "engineering.sqlite3"
DB_PATH = _DB_PATH


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class WorkCentre:
    id: int
    name: str
    code: str | None
    description: str
    notes: str
    created_at: str
    updated_at: str

    def serialise(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "description": self.description,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class WorkCentreStore:
    """SQLite-backed persistence for work centres."""

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
                CREATE TABLE IF NOT EXISTS work_centres (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    code TEXT UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def list(self, *, name_contains: str | None = None) -> list[WorkCentre]:
        query = "SELECT * FROM work_centres"
        params: Sequence[object]
        if name_contains:
            query += " WHERE name LIKE ?"
            params = (f"%{name_contains}%",)
        else:
            params = ()
        query += " ORDER BY name"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_model(row) for row in rows]

    def get(self, work_centre_id: int) -> WorkCentre | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM work_centres WHERE id = ?",
                (work_centre_id,),
            ).fetchone()
        return self._row_to_model(row) if row else None

    def create(
        self,
        *,
        name: str,
        code: str | None = None,
        description: str = "",
        notes: str = "",
    ) -> WorkCentre:
        name = name.strip()
        if not name:
            raise ValueError("Work centre name is required.")
        code = code.strip() if code else None
        description = description.strip()
        notes = notes.strip()
        timestamp = _utcnow()
        with self._lock, self._connect() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO work_centres (name, code, description, notes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (name, code, description, notes, timestamp, timestamp),
                )
            except sqlite3.IntegrityError as exc:  # pragma: no cover - handled below
                raise _map_integrity_error(exc) from exc
            work_centre_id = cursor.lastrowid
        result = self.get(work_centre_id)
        if result is None:  # pragma: no cover - sanity guard
            raise RuntimeError("Failed to load work centre after creation.")
        return result

    def update(self, work_centre_id: int, **fields: object) -> WorkCentre:
        allowed = {"name", "code", "description", "notes"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            raise ValueError("No valid fields provided for update.")
        if "name" in updates:
            name = str(updates["name"]).strip()
            if not name:
                raise ValueError("Work centre name is required.")
            updates["name"] = name
        if "code" in updates:
            code = str(updates["code"]).strip()
            updates["code"] = code or None
        if "description" in updates:
            updates["description"] = str(updates["description"]).strip()
        if "notes" in updates:
            updates["notes"] = str(updates["notes"]).strip()
        assignments = ", ".join(f"{field} = ?" for field in updates)
        params: list[object] = [updates[field] for field in updates]
        params.append(_utcnow())
        params.append(work_centre_id)
        with self._lock, self._connect() as conn:
            try:
                updated = conn.execute(
                    f"UPDATE work_centres SET {assignments}, updated_at = ? WHERE id = ?",
                    params,
                )
            except sqlite3.IntegrityError as exc:  # pragma: no cover - handled below
                raise _map_integrity_error(exc) from exc
            if updated.rowcount == 0:
                raise KeyError(work_centre_id)
        result = self.get(work_centre_id)
        if result is None:  # pragma: no cover - sanity guard
            raise KeyError(work_centre_id)
        return result

    @staticmethod
    def _row_to_model(row: sqlite3.Row | None) -> WorkCentre:
        assert row is not None
        return WorkCentre(
            id=int(row["id"]),
            name=str(row["name"]),
            code=row["code"],
            description=str(row["description"] or ""),
            notes=str(row["notes"] or ""),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )


def _map_integrity_error(exc: sqlite3.IntegrityError) -> ValueError:
    message = str(exc).lower()
    if "work_centres.name" in message:
        return ValueError("A work centre with this name already exists.")
    if "work_centres.code" in message:
        return ValueError("A work centre with this code already exists.")
    return ValueError("Unable to persist work centre.")


WORK_CENTRES = WorkCentreStore()


def reset_work_centres() -> None:
    WORK_CENTRES._initialise()  # type: ignore[attr-defined]
