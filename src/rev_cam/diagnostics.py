"""Command-line helpers for RevCam diagnostics."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Sequence

from .camera import (
    NUMPY_ABI_HINT,
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

PICAMERA_REINSTALL_HINT = (
    "Reinstall the Raspberry Pi OS Picamera2 stack (`sudo apt install --reinstall "
    "python3-picamera2 python3-simplejpeg`; SimpleJPEG is only packaged with the "
    "`python3-` prefix)."
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

    print(f"RevCam diagnostics (version {APP_VERSION})")
    if hints:
        print("Detected potential PiCamera2 conflicts:")
        for hint in hints:
            print(f" - {hint}")
    else:
        print("No conflicting services or processes were detected.")
        print(
            "If the camera still reports 'Device or resource busy', double-check"
            " for external processes or legacy camera settings."
        )

    if picamera_stack["status"] == "ok":
        print("Picamera2 Python stack: OK")
        if "numpy_version" in picamera_stack:
            print(f" - NumPy version: {picamera_stack['numpy_version']}")
    else:
        print("Picamera2 Python stack issues detected:")
        for detail in picamera_stack["details"]:
            print(f" - {detail}")
        hints_payload = picamera_stack.get("hints")
        if hints_payload:
            print("Hints:")
            for hint in hints_payload:
                print(f" * {hint}")
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
