"""Mode management for mutually exclusive reversing and surveillance modes."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Callable, Mapping

import json


ModeName = str


@dataclass(slots=True)
class ModeController:
    """Lifecycle callbacks for a single mode."""

    start: Callable[[], None] | None = None
    stop: Callable[[], None] | None = None


class ModeSwitchError(RuntimeError):
    """Raised when a requested mode transition cannot be satisfied."""


class ModeManager:
    """Coordinate exclusive access to shared hardware between operating modes."""

    VALID_MODES: tuple[ModeName, ...] = ("idle", "reversing", "surveillance")

    def __init__(
        self,
        *,
        lock_path: Path | str = Path("data/camera.lock"),
        state_path: Path | str = Path("data/mode_state.json"),
        controllers: Mapping[ModeName, ModeController] | None = None,
    ) -> None:
        self._lock_path = Path(lock_path)
        self._state_path = Path(state_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._controllers = dict(controllers or {})
        self._mutex = Lock()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _read_state(self) -> ModeName:
        if not self._state_path.exists():
            return "idle"
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - defensive branch
            return "idle"
        mode = raw.get("mode", "idle")
        if mode not in self.VALID_MODES:
            return "idle"
        return mode

    def _write_state(self, mode: ModeName) -> None:
        payload = {
            "mode": mode,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_lock(self, mode: ModeName) -> None:
        if mode == "idle":
            if self._lock_path.exists():
                self._lock_path.unlink()
            return
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.write_text(mode, encoding="utf-8")

    def _invoke_controller(self, previous: ModeName, new_mode: ModeName) -> None:
        if previous == new_mode:
            return
        prev_controller = self._controllers.get(previous)
        if prev_controller and prev_controller.stop:
            prev_controller.stop()
        next_controller = self._controllers.get(new_mode)
        if next_controller and next_controller.start:
            next_controller.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def current_mode(self) -> ModeName:
        """Return the currently active mode recorded on disk."""

        with self._mutex:
            return self._read_state()

    def set_mode(self, requested: ModeName) -> ModeName:
        """Attempt to switch to *requested* mode and return the active mode."""

        if requested not in self.VALID_MODES:
            raise ModeSwitchError(f"Unknown mode {requested!r}")

        with self._mutex:
            current = self._read_state()
            if requested == current:
                return current
            if requested != "idle" and self._lock_path.exists():
                lock_owner = self._lock_path.read_text(encoding="utf-8").strip()
                if lock_owner and lock_owner != requested:
                    raise ModeSwitchError(
                        f"Camera already in use by {lock_owner}; stop it before switching to {requested}"
                    )
            self._write_lock(requested)
            try:
                self._invoke_controller(current, requested)
            except Exception as exc:  # pragma: no cover - defensive branch
                # Attempt to revert lock state before bubbling error
                self._write_lock(current)
                raise ModeSwitchError(str(exc)) from exc
            self._write_state(requested)
            return requested

    def clear_mode(self) -> None:
        """Transition to idle state and release the lock."""

        with self._mutex:
            current = self._read_state()
            if current != "idle":
                self._invoke_controller(current, "idle")
            self._write_lock("idle")
            self._write_state("idle")


__all__ = ["ModeController", "ModeManager", "ModeSwitchError"]
