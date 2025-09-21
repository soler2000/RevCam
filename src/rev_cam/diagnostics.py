"""Command-line helpers for RevCam diagnostics."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Sequence

from .camera import (
    NUMPY_ABI_HINT,
    PICAMERA_REINSTALL_HINT,
    detect_numpy_abi_mismatch,
    diagnose_camera_conflicts,
    summarise_exception,
)
from .version import APP_VERSION

PICAMERA_INSTALL_HINT = (
    "Install the Raspberry Pi OS Picamera2 packages with `sudo apt install python3-picamera2` "
    "and recreate the virtual environment with `python3 -m venv --system-site-packages .venv` "
    "or run `./scripts/install.sh --pi`."
)

NUMPY_INSTALL_HINT = (
    "Install NumPy inside the active environment (for example `pip install numpy` or rerun "
    "`./scripts/install.sh --pi`)."
)

def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the diagnostics CLI."""

    parser = argparse.ArgumentParser(
        prog="python -m rev_cam.diagnostics",
        description="RevCam diagnostics helpers",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON for scripting.",
    )
    return parser


def diagnose_picamera_stack() -> dict[str, object]:
    """Return diagnostic details about the Picamera2 Python stack."""

    status = "ok"
    details: list[str] = []
    hints: list[str] = []
    numpy_version: str | None = None

    def mark_error(detail: str) -> None:
        nonlocal status
        status = "error"
        details.append(detail)

    def add_hint(text: str) -> None:
        if text not in hints:
            hints.append(text)

    try:
        numpy_module = importlib.import_module("numpy")
    except ModuleNotFoundError:
        mark_error("numpy module not found.")
        add_hint(NUMPY_INSTALL_HINT)
    except Exception as exc:  # pragma: no cover - defensive guard
        detail = summarise_exception(exc)
        mark_error(f"numpy import failed: {detail}")
        if detect_numpy_abi_mismatch(detail):
            add_hint(NUMPY_ABI_HINT)
            add_hint(PICAMERA_REINSTALL_HINT)
    else:
        numpy_version = getattr(numpy_module, "__version__", None)

    try:
        importlib.import_module("picamera2")
    except ModuleNotFoundError:
        mark_error("picamera2 module not found.")
        add_hint(PICAMERA_INSTALL_HINT)
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        detail = summarise_exception(exc)
        mark_error(f"picamera2 import failed: {detail}")
        lower_detail = detail.lower()
        if detect_numpy_abi_mismatch(detail):
            add_hint(NUMPY_ABI_HINT)
            add_hint(PICAMERA_REINSTALL_HINT)
        if "simplejpeg" in lower_detail:
            add_hint(PICAMERA_REINSTALL_HINT)
    else:
        # Import succeeded, nothing else to record
        pass

    payload: dict[str, object] = {"status": status, "details": details}
    if hints:
        payload["hints"] = hints
    if numpy_version:
        payload["numpy_version"] = numpy_version
    return payload


def collect_diagnostics() -> dict[str, object]:
    """Collect diagnostics payload used by both the CLI and API."""

    hints = diagnose_camera_conflicts()
    picamera_stack = diagnose_picamera_stack()
    return {
        "version": APP_VERSION,
        "camera_conflicts": hints,
        "picamera": picamera_stack,
    }


def run(argv: Sequence[str] | None = None) -> int:
    """Execute the diagnostics CLI with *argv* arguments."""

    parser = build_parser()
    args = parser.parse_args(argv)

    payload = collect_diagnostics()

    if args.json:
        json.dump(payload, sys.stdout)
        sys.stdout.write("\n")
        return 0

    camera_conflicts = list(payload.get("camera_conflicts") or [])
    picamera_stack = payload.get("picamera") or {}
    picamera_status = picamera_stack.get("status")
    picamera_details = list(picamera_stack.get("details") or [])
    picamera_hints = list(picamera_stack.get("hints") or [])
    numpy_version = picamera_stack.get("numpy_version")

    print(f"RevCam diagnostics (version {APP_VERSION})")
    if camera_conflicts:
        print("Detected potential PiCamera2 conflicts:")
        for hint in camera_conflicts:
            print(f" - {hint}")
    else:
        print("No conflicting services or processes were detected.")
        print(
            "If the camera still reports 'Device or resource busy', double-check"
            " for external processes or legacy camera settings."
        )

    if picamera_status == "ok":
        print("Picamera2 Python stack: OK")
        if numpy_version:
            print(f" - NumPy version: {numpy_version}")
    elif picamera_status == "error":
        print("Picamera2 Python stack issues detected:")
        for detail in picamera_details:
            print(f" - {detail}")
        if picamera_hints:
            print("Hints:")
            for hint in picamera_hints:
                print(f" * {hint}")
    else:
        print(f"Picamera2 Python stack status: {picamera_status or 'unknown'}")
        for detail in picamera_details:
            print(f" - {detail}")
        if picamera_hints:
            print("Hints:")
            for hint in picamera_hints:
                print(f" * {hint}")
        if numpy_version:
            print(f" - NumPy version: {numpy_version}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by `python -m rev_cam.diagnostics`."""

    return run(argv)


__all__ = [
    "build_parser",
    "diagnose_picamera_stack",
    "collect_diagnostics",
    "run",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - module behaviour
    sys.exit(main())
