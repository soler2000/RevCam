"""General-purpose persistent event log for RevCam components."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable


@dataclass(slots=True)
class SystemLogEntry:
    """Represents a system event captured for troubleshooting."""

    timestamp: float
    category: str
    event: str
    message: str
    status: dict[str, object | None] | None = None
    metadata: dict[str, object | None] | None = None

    def to_dict(self) -> dict[str, object | None]:
        payload: dict[str, object | None] = {
            "timestamp": self.timestamp,
            "category": self.category,
            "event": self.event,
            "message": self.message,
        }
        if self.status is not None:
            payload["status"] = self.status
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class SystemLog:
    """Persistent append-only log shared across subsystems."""

    def __init__(
        self,
        path: Path | str | None = Path("data/system_log.jsonl"),
        *,
        max_entries: int = 500,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._path: Path | None = Path(path) if path is not None else None
        self._entries: Deque[SystemLogEntry] = deque(maxlen=max_entries)
        self._lock = threading.Lock()
        if self._path is not None:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # pragma: no cover - filesystem errors are rare
                logging.getLogger(__name__).warning(
                    "Unable to prepare system log directory: %s", exc
                )
                self._path = None
        self._load_entries()

    # ------------------------------ properties -----------------------------
    @property
    def path(self) -> Path | None:
        """Return the backing file path when persistence is enabled."""

        return self._path

    # ------------------------------ operations -----------------------------
    def record(
        self,
        category: str,
        event: str,
        message: str,
        *,
        status: dict[str, object | None] | None = None,
        metadata: dict[str, object | None] | None = None,
    ) -> SystemLogEntry:
        """Append a new event to the log and return the stored entry."""

        cleaned_category = category.strip() if isinstance(category, str) else ""
        if not cleaned_category:
            cleaned_category = "general"
        entry = SystemLogEntry(
            timestamp=time.time(),
            category=cleaned_category,
            event=event,
            message=message,
            status=status,
            metadata=self._clean_metadata(metadata),
        )
        with self._lock:
            self._entries.append(entry)
            self._append_persistent(entry)
        return entry

    def tail(
        self,
        limit: int | None = None,
        *,
        category: str | None = None,
    ) -> list[SystemLogEntry]:
        """Return the most recent entries, optionally filtering by category."""

        with self._lock:
            entries: Iterable[SystemLogEntry] = list(self._entries)
        if category is not None:
            wanted = category.strip()
            if wanted:
                entries = [entry for entry in entries if entry.category == wanted]
            else:
                entries = list(entries)
        if limit is not None:
            try:
                limit_value = max(1, int(limit))
            except (TypeError, ValueError):
                limit_value = 1
            if len(entries) > limit_value:
                entries = entries[-limit_value:]
        return list(entries)

    # ----------------------------- implementation --------------------------
    def _load_entries(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError as exc:  # pragma: no cover - best effort logging
            logging.getLogger(__name__).warning("Unable to load system log: %s", exc)
            return
        restored: Deque[SystemLogEntry] = deque(maxlen=self._entries.maxlen)
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            entry = self._deserialize(payload)
            if entry is not None:
                restored.append(entry)
        if restored:
            with self._lock:
                for entry in restored:
                    self._entries.append(entry)

    def _deserialize(self, payload: object) -> SystemLogEntry | None:
        if not isinstance(payload, dict):
            return None
        event = payload.get("event")
        message = payload.get("message")
        category = payload.get("category")
        timestamp = payload.get("timestamp")
        if not isinstance(event, str) or not isinstance(message, str):
            return None
        cleaned_category = category.strip() if isinstance(category, str) else "general"
        try:
            ts_value = float(timestamp) if timestamp is not None else time.time()
        except (TypeError, ValueError):
            ts_value = time.time()
        status_payload = payload.get("status")
        if not isinstance(status_payload, dict):
            status_payload = None
        metadata_payload = payload.get("metadata")
        if not isinstance(metadata_payload, dict):
            metadata_payload = None
        return SystemLogEntry(
            timestamp=ts_value,
            category=cleaned_category,
            event=event,
            message=message,
            status=status_payload,
            metadata=metadata_payload,
        )

    def _append_persistent(self, entry: SystemLogEntry) -> None:
        if self._path is None:
            return
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")
        except OSError as exc:  # pragma: no cover - best effort logging
            logging.getLogger(__name__).warning("Unable to persist system log: %s", exc)

    @staticmethod
    def _clean_metadata(
        metadata: dict[str, object | None] | None,
    ) -> dict[str, object | None] | None:
        if not metadata:
            return None
        cleaned: dict[str, object | None] = {}
        for key, value in metadata.items():
            if value is not None:
                cleaned[key] = value
        return cleaned or None


__all__ = ["SystemLog", "SystemLogEntry"]
