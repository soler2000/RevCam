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
    "`python3-` prefix). If APT cannot find SimpleJPEG, install it inside the "
    "RevCam virtual environment with `pip install --prefer-binary simplejpeg` (add "
    "the PiWheels index on Raspberry Pi)."
)

WEBRTC_PREREQS_HINT = (
    "Install the native WebRTC prerequisites with `sudo ./scripts/install_prereqs.sh` "
    "before recreating the RevCam virtual environment."
)

WEBRTC_PIP_HINT = (
    "Ensure PyAV and aiortc are installed inside the active environment by running "
    "`./scripts/install.sh --pi` (or `pip install av aiortc`)."
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


def diagnose_webrtc_stack() -> dict[str, object]:
    """Return diagnostic details about the WebRTC software stack."""

    status = "ok"
    details: list[str] = []
    hints: list[str] = []

    def mark_error(detail: str) -> None:
        nonlocal status
        status = "error"
        details.append(detail)

    def add_hint(text: str) -> None:
        if text not in hints:
            hints.append(text)

    modules = (
        ("av", "PyAV"),
        ("aiortc", "aiortc"),
    )

    for module_name, friendly in modules:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            mark_error(f"{friendly} module not found.")
            add_hint(WEBRTC_PREREQS_HINT)
            add_hint(WEBRTC_PIP_HINT)
            continue
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            detail = summarise_exception(exc)
            mark_error(f"{friendly} import failed: {detail}")
            add_hint(WEBRTC_PREREQS_HINT)
            add_hint(WEBRTC_PIP_HINT)
            continue

        if module_name == "av" and not hasattr(module, "VideoFrame"):
            mark_error("PyAV VideoFrame support unavailable.")
            add_hint(WEBRTC_PIP_HINT)

        if module_name == "aiortc":
            try:
                importlib.import_module("aiortc.rtcrtpsender")
            except Exception as exc:  # pragma: no cover - depends on runtime env
                detail = summarise_exception(exc)
                mark_error(f"aiortc runtime import failed: {detail}")
                add_hint(WEBRTC_PREREQS_HINT)
                add_hint(WEBRTC_PIP_HINT)

    payload: dict[str, object] = {"status": status, "details": details}
    if hints:
        payload["hints"] = hints
    return payload


def collect_diagnostics() -> dict[str, object]:
    """Collect diagnostics payload used by both the CLI and API."""

    hints = diagnose_camera_conflicts()
    picamera_stack = diagnose_picamera_stack()
    webrtc_stack = diagnose_webrtc_stack()
    return {
        "version": APP_VERSION,
        "camera_conflicts": hints,
        "picamera": picamera_stack,
        "webrtc": webrtc_stack,
    }


def run(argv: Sequence[str] | None = None) -> int:
    """Execute the diagnostics CLI with *argv* arguments."""

    parser = build_parser()
    args = parser.parse_args(argv)

    payload = collect_diagnostics()
    camera_conflicts = payload.get("camera_conflicts", [])
    picamera_stack = payload.get("picamera", {})
    webrtc_stack = payload.get("webrtc", {})

    if not isinstance(camera_conflicts, list):
        camera_conflicts = list(camera_conflicts) if camera_conflicts else []

    if not isinstance(picamera_stack, dict):
        picamera_stack = {}
    if not isinstance(webrtc_stack, dict):
        webrtc_stack = {}

    if args.json:
        json.dump(payload, sys.stdout)
        sys.stdout.write("\n")
        return 0

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

    status = picamera_stack.get("status")

    if status == "ok":
        print("Picamera2 Python stack: OK")
        if "numpy_version" in picamera_stack:
            print(f" - NumPy version: {picamera_stack['numpy_version']}")
    else:
        print("Picamera2 Python stack issues detected:")
        for detail in picamera_stack.get("details", []):
            print(f" - {detail}")
        hints_payload = picamera_stack.get("hints")
        if hints_payload:
            print("Hints:")
            for hint in hints_payload:
                print(f" * {hint}")
    webrtc_status = webrtc_stack.get("status")

    if webrtc_status == "ok":
        print("WebRTC stack: OK")
    else:
        print("WebRTC stack issues detected:")
        for detail in webrtc_stack.get("details", []):
            print(f" - {detail}")
        hints_payload = webrtc_stack.get("hints")
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
    "diagnose_webrtc_stack",
    "collect_diagnostics",
    "run",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - module behaviour
    sys.exit(main())
