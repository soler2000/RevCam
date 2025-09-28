"""High level management of surveillance clips and settings."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Iterable, Mapping, Sequence

import base64
import hashlib
import json
import sqlite3
import uuid
import zipfile

from fastapi import HTTPException

from .settings import SurveillanceSettings, SurveillanceSettingsStore


@dataclass(slots=True)
class ClipRecord:
    id: int
    path: str
    start_ts: datetime
    end_ts: datetime
    duration_s: float
    size_bytes: int
    has_audio: bool
    thumb_path: str | None
    motion_score: float | None
    settings_hash: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["start_ts"] = self.start_ts.isoformat()
        payload["end_ts"] = self.end_ts.isoformat()
        return payload


@dataclass(slots=True)
class ClipFilters:
    """Query parameters for clip searches."""

    from_ts: datetime | None = None
    to_ts: datetime | None = None
    page: int = 1
    page_size: int = 50
    sort: str = "-start_ts"


class SurveillanceManager:
    """Entry point coordinating storage and metadata management."""

    def __init__(
        self,
        *,
        base_path: Path | str = Path("data/surveillance"),
        settings_store: SurveillanceSettingsStore | None = None,
    ) -> None:
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._db_path = self._base_path / "index.db"
        self._exports_path = self._base_path / "exports"
        self._exports_path.mkdir(parents=True, exist_ok=True)
        self._settings_store = (
            settings_store
            if settings_store is not None
            else SurveillanceSettingsStore(self._base_path / "settings.json")
        )
        self._mutex = RLock()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def load_settings(self) -> SurveillanceSettings:
        return self._settings_store.load()

    def save_settings(self, payload: dict) -> SurveillanceSettings:
        settings = self.load_settings()
        settings = SurveillanceSettings.from_dict({**settings.to_dict(), **payload})
        self._settings_store.save(settings)
        return settings

    @property
    def base_path(self) -> Path:
        return self._base_path

    @property
    def exports_path(self) -> Path:
        return self._exports_path

    # ------------------------------------------------------------------
    # Manual recording
    # ------------------------------------------------------------------
    def request_manual_record(
        self,
        *,
        duration_s: float | int | None = None,
    ) -> dict[str, object]:
        if duration_s is not None:
            try:
                duration_val = float(duration_s)
            except (TypeError, ValueError):
                raise ValueError("Manual record duration must be numeric") from None
            if duration_val <= 0:
                raise ValueError("Manual record duration must be positive")
            if duration_val > 3600:
                raise ValueError("Manual record duration cannot exceed one hour")
        else:
            duration_val = None

        clip = self.create_manual_clip(duration_val)
        request_id = uuid.uuid4().hex[:12]
        requested_at = clip.start_ts
        payload = {
            "id": request_id,
            "requested_at": requested_at.isoformat(),
            "duration_s": clip.duration_s,
            "clip_id": clip.id,
        }
        self._write_manual_request(payload)
        return {
            "id": request_id,
            "requested_at": requested_at,
            "duration_s": clip.duration_s,
            "clip": clip,
        }

    def create_manual_clip(self, duration_s: float | None) -> ClipRecord:
        settings = self.load_settings()
        duration = float(duration_s) if duration_s is not None else float(min(30, settings.clip_max_length_s))
        if duration <= 0:
            raise ValueError("Manual clip duration must be positive")
        if duration > settings.clip_max_length_s:
            duration = float(settings.clip_max_length_s)
        end_ts = datetime.now(timezone.utc)
        start_ts = end_ts - timedelta(seconds=duration)
        return self._create_placeholder_clip(
            start_ts=start_ts,
            end_ts=end_ts,
            has_audio=settings.audio_enabled,
            motion_score=None,
            reason="manual",
        )

    def create_motion_clip(
        self,
        *,
        start_ts: datetime,
        end_ts: datetime,
        motion_score: float | None,
    ) -> ClipRecord:
        settings = self.load_settings()
        return self._create_placeholder_clip(
            start_ts=start_ts,
            end_ts=end_ts,
            has_audio=settings.audio_enabled,
            motion_score=motion_score,
            reason="motion",
        )

    # ------------------------------------------------------------------
    # Clip operations
    # ------------------------------------------------------------------
    def register_clip(
        self,
        *,
        start_ts: datetime,
        end_ts: datetime,
        path: Path,
        thumb_path: Path | None,
        has_audio: bool,
        motion_score: float | None,
        settings: SurveillanceSettings,
    ) -> ClipRecord:
        size_bytes = path.stat().st_size if path.exists() else 0
        record = ClipRecord(
            id=-1,
            path=str(path),
            start_ts=start_ts,
            end_ts=end_ts,
            duration_s=(end_ts - start_ts).total_seconds(),
            size_bytes=size_bytes,
            has_audio=has_audio,
            thumb_path=str(thumb_path) if thumb_path else None,
            motion_score=motion_score,
            settings_hash=self._hash_settings(settings),
        )
        with self._mutex:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO clips (
                        path, start_ts, end_ts, duration_s, size_bytes,
                        has_audio, thumb_path, motion_score, settings_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.path,
                        record.start_ts.isoformat(),
                        record.end_ts.isoformat(),
                        record.duration_s,
                        record.size_bytes,
                        1 if record.has_audio else 0,
                        record.thumb_path,
                        record.motion_score,
                        record.settings_hash,
                    ),
                )
                conn.commit()
            record = ClipRecord(
                id=int(cursor.lastrowid),
                path=record.path,
                start_ts=record.start_ts,
                end_ts=record.end_ts,
                duration_s=record.duration_s,
                size_bytes=record.size_bytes,
                has_audio=record.has_audio,
                thumb_path=record.thumb_path,
                motion_score=record.motion_score,
                settings_hash=record.settings_hash,
            )
        return record

    def list_clips(self, filters: ClipFilters) -> tuple[list[ClipRecord], int]:
        query = "SELECT id, path, start_ts, end_ts, duration_s, size_bytes, has_audio, thumb_path, motion_score, settings_hash FROM clips"
        clauses: list[str] = []
        params: list[object] = []
        if filters.from_ts is not None:
            clauses.append("start_ts >= ?")
            params.append(filters.from_ts.isoformat())
        if filters.to_ts is not None:
            clauses.append("start_ts <= ?")
            params.append(filters.to_ts.isoformat())
        filter_clause = ""
        if clauses:
            filter_clause = " WHERE " + " AND ".join(clauses)
            query += filter_clause
        order = filters.sort or "-start_ts"
        direction = "DESC" if order.startswith("-") else "ASC"
        field = order.lstrip("+-")
        if field not in {"start_ts", "duration_s", "size_bytes", "motion_score"}:
            field = "start_ts"
        query += f" ORDER BY {field} {direction}"
        limit = max(1, min(int(filters.page_size), 500))
        offset = max(0, (int(filters.page) - 1) * limit)
        query += " LIMIT ? OFFSET ?"
        filter_params = list(params)
        params.extend([limit, offset])
        with self._mutex:
            with self._connect() as conn:
                rows = conn.execute(query, params).fetchall()
                total = conn.execute(
                    f"SELECT COUNT(*) FROM clips{filter_clause}",
                    filter_params,
                ).fetchone()[0]
        records = [self._row_to_clip(row) for row in rows]
        return records, int(total)

    def get_clip(self, clip_id: int) -> ClipRecord:
        with self._mutex:
            with self._connect() as conn:
                row = conn.execute(
                """
                SELECT id, path, start_ts, end_ts, duration_s, size_bytes,
                       has_audio, thumb_path, motion_score, settings_hash
                FROM clips WHERE id = ?
                """,
                (int(clip_id),),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Clip not found")
        return self._row_to_clip(row)

    def delete_clips(self, clip_ids: Iterable[int]) -> int:
        clip_ids = [int(clip_id) for clip_id in clip_ids]
        with self._mutex:
            with self._connect() as conn:
                return self._delete_ids(conn, clip_ids)

    def export_clips(self, clip_ids: Sequence[int]) -> Path:
        if not clip_ids:
            raise HTTPException(status_code=400, detail="No clip IDs supplied")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        export_path = self._exports_path / f"export-{timestamp}-{uuid.uuid4().hex[:8]}.zip"
        with self._mutex:
            with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_STORED) as archive:
                for clip_id in clip_ids:
                    record = self.get_clip(int(clip_id))
                    path = Path(record.path)
                    if not path.exists():
                        continue
                    archive.write(path, arcname=path.name)
                    if record.thumb_path:
                        thumb = Path(record.thumb_path)
                        if thumb.exists():
                            archive.write(thumb, arcname=thumb.name)
        return export_path

    def apply_retention(self) -> int:
        settings = self.load_settings()
        removed = 0
        with self._mutex:
            with self._connect() as conn:
                if settings.storage_max_days is not None:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=int(settings.storage_max_days))
                    rows = conn.execute(
                        "SELECT id FROM clips WHERE start_ts < ?",
                        (cutoff.isoformat(),),
                    ).fetchall()
                    removed += self._delete_ids(conn, [int(row[0]) for row in rows])
                if settings.storage_max_size_gb is not None:
                    limit_bytes = int(float(settings.storage_max_size_gb) * 1024**3)
                    rows = conn.execute(
                        "SELECT id, size_bytes FROM clips ORDER BY start_ts ASC",
                    ).fetchall()
                    total = sum(int(row[1]) for row in rows)
                    for clip_id, size in rows:
                        if total <= limit_bytes:
                            break
                        removed += self._delete_ids(conn, [int(clip_id)])
                        total -= int(size)
                conn.commit()
        return removed

    def ensure_day_folder(self, when: datetime) -> Path:
        folder = self._base_path / when.strftime("%Y/%m/%d")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def create_test_clip(self) -> ClipRecord:
        """Create a placeholder clip for development and testing."""

        settings = self.load_settings()
        now = datetime.now(timezone.utc)
        start_ts = now - timedelta(seconds=8)
        end_ts = now
        return self._create_placeholder_clip(
            start_ts=start_ts,
            end_ts=end_ts,
            has_audio=settings.audio_enabled,
            motion_score=1.0,
            reason="test",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL,
                    start_ts TEXT NOT NULL,
                    end_ts TEXT NOT NULL,
                    duration_s REAL NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    has_audio INTEGER NOT NULL,
                    thumb_path TEXT,
                    motion_score REAL,
                    settings_hash TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_clips_start_ts ON clips(start_ts)")

    def _delete_ids(self, conn: sqlite3.Connection, clip_ids: Sequence[int]) -> int:
        removed = 0
        for clip_id in clip_ids:
            clip = conn.execute(
                "SELECT path, thumb_path FROM clips WHERE id = ?",
                (int(clip_id),),
            ).fetchone()
            if clip is None:
                continue
            path, thumb_path = clip
            conn.execute("DELETE FROM clips WHERE id = ?", (int(clip_id),))
            removed += 1
            if path and Path(path).exists():
                Path(path).unlink()
            if thumb_path and Path(thumb_path).exists():
                Path(thumb_path).unlink()
        conn.commit()
        return removed

    def _row_to_clip(self, row: sqlite3.Row) -> ClipRecord:
        return ClipRecord(
            id=int(row["id"]),
            path=str(row["path"]),
            start_ts=datetime.fromisoformat(row["start_ts"]),
            end_ts=datetime.fromisoformat(row["end_ts"]),
            duration_s=float(row["duration_s"]),
            size_bytes=int(row["size_bytes"]),
            has_audio=bool(row["has_audio"]),
            thumb_path=row["thumb_path"],
            motion_score=row["motion_score"] if row["motion_score"] is not None else None,
            settings_hash=str(row["settings_hash"]),
        )

    def _hash_settings(self, settings: SurveillanceSettings) -> str:
        payload = json.dumps(settings.to_dict(), sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return digest[:16]

    def _write_manual_request(self, payload: Mapping[str, object]) -> None:
        folder = self._base_path / "manual_requests"
        folder.mkdir(parents=True, exist_ok=True)
        file_name = payload.get("requested_at", datetime.now(timezone.utc).isoformat())
        request_id = payload.get("id", uuid.uuid4().hex[:12])
        stem = str(file_name).replace(":", "-")
        path = folder / f"manual-{stem}-{request_id}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _create_placeholder_clip(
        self,
        *,
        start_ts: datetime,
        end_ts: datetime,
        has_audio: bool,
        motion_score: float | None,
        reason: str,
    ) -> ClipRecord:
        folder = self.ensure_day_folder(end_ts)
        iso = end_ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        clip_id = uuid.uuid4().hex[:8]
        clip_name = f"clip-{iso}-{clip_id}.mp4"
        clip_path = folder / clip_name
        summary = (
            f"RevCam placeholder clip generated for {reason} event\n"
            f"Start: {start_ts.isoformat()}\nEnd: {end_ts.isoformat()}\n"
        )
        clip_path.write_text(summary, encoding="utf-8")
        thumb_path = folder / f"clip-{iso}-{clip_id}.jpg"
        thumb_path.write_bytes(_PLACEHOLDER_JPEG)
        settings = self.load_settings()
        return self.register_clip(
            start_ts=start_ts,
            end_ts=end_ts,
            path=clip_path,
            thumb_path=thumb_path,
            has_audio=has_audio,
            motion_score=motion_score,
            settings=settings,
        )


_PLACEHOLDER_JPEG = bytes(
    [
        0xFF,
        0xD8,
        0xFF,
        0xE0,
        0x00,
        0x10,
        0x4A,
        0x46,
        0x49,
        0x46,
        0x00,
        0x01,
        0x01,
        0x00,
        0x00,
        0x01,
        0x00,
        0x01,
        0x00,
        0x00,
        0xFF,
        0xDB,
        0x00,
        0x43,
        0x00,
        0x08,
        0x06,
        0x06,
        0x07,
        0x06,
        0x05,
        0x08,
        0x07,
        0x07,
        0x07,
        0x09,
        0x09,
        0x08,
        0x0A,
        0x0C,
        0x14,
        0x0D,
        0x0C,
        0x0B,
        0x0B,
        0x0C,
        0x19,
        0x12,
        0x13,
        0x0F,
        0x14,
        0x1D,
        0x1A,
        0x1F,
        0x1E,
        0x1D,
        0x1A,
        0x1C,
        0x1C,
        0x20,
        0x24,
        0x2E,
        0x27,
        0x20,
        0x28,
        0x23,
        0x1C,
        0x1C,
        0x2B,
        0x37,
        0x2B,
        0x28,
        0x30,
        0x34,
        0x39,
        0x3A,
        0x3F,
        0x3F,
        0x3F,
        0x1F,
        0x27,
        0x39,
        0x3D,
        0x3F,
        0x34,
        0x2E,
        0x37,
        0x38,
        0xFF,
        0xC0,
        0x00,
        0x0B,
        0x08,
        0x00,
        0x01,
        0x00,
        0x01,
        0x01,
        0x01,
        0x11,
        0x00,
        0xFF,
        0xC4,
        0x00,
        0x14,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0xFF,
        0xC4,
        0x00,
        0x14,
        0x10,
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0xFF,
        0xDA,
        0x00,
        0x08,
        0x01,
        0x01,
        0x00,
        0x00,
        0x3F,
        0x00,
        0xD2,
        0xCF,
        0x20,
        0xFF,
        0xD9,
    ]
)


__all__ = ["ClipFilters", "ClipRecord", "SurveillanceManager"]
