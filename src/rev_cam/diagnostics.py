"""Command-line helpers for RevCam diagnostics."""
from __future__ import annotations

import argparse
import importlib
import json
import os
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


def _clamp_percentage(value: float) -> float:
    """Clamp *value* to the 0–100 range."""

    return max(0.0, min(100.0, value))


def _collect_cpu_metrics() -> dict[str, object] | None:
    """Return CPU utilisation metrics if available."""

    metrics: dict[str, object] = {}

    cpu_count = os.cpu_count()
    if isinstance(cpu_count, int) and cpu_count > 0:
        metrics["count"] = cpu_count
    else:
        cpu_count = None

    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except (AttributeError, OSError):  # pragma: no cover - depends on OS support
        load_1m = load_5m = load_15m = None
    else:
        metrics["load"] = {"1m": load_1m, "5m": load_5m, "15m": load_15m}
        if cpu_count:
            metrics["usage_percent"] = _clamp_percentage((load_1m / cpu_count) * 100.0)

    per_core_metrics = _collect_per_core_cpu_usage()
    if per_core_metrics:
        metrics["per_core"] = per_core_metrics

    return metrics or None


def _collect_per_core_cpu_usage() -> list[dict[str, object]]:
    """Return per-core CPU utilisation percentages if available."""

    try:
        with open("/proc/stat", "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:  # pragma: no cover - defensive guard for non-Linux platforms
        return []

    per_core: list[dict[str, object]] = []

    for line in lines:
        if not line.startswith("cpu"):
            continue

        parts = line.split()
        if not parts:
            continue

        label = parts[0]
        if label == "cpu":
            # Skip the aggregated entry – load averages already summarise overall usage.
            continue

        numeric_parts: list[int] = []
        for value in parts[1:]:
            try:
                numeric_parts.append(int(value))
            except ValueError:
                numeric_parts.append(0)

        if len(numeric_parts) < 4:
            continue

        total_time = sum(numeric_parts[:8]) if len(numeric_parts) >= 8 else sum(numeric_parts)
        if total_time <= 0:
            continue

        idle_time = numeric_parts[3]
        if len(numeric_parts) > 4:
            idle_time += numeric_parts[4]

        busy_time = max(0, total_time - idle_time)
        usage_percent = _clamp_percentage((busy_time / total_time) * 100.0)

        entry: dict[str, object] = {"usage_percent": usage_percent, "label": label}

        if label.startswith("cpu"):
            suffix = label[3:]
            if suffix:
                try:
                    index = int(suffix)
                except ValueError:
                    index = None
                else:
                    if index >= 0:
                        entry["index"] = index

        per_core.append(entry)

    per_core.sort(
        key=lambda item: (
            item.get("index") if isinstance(item.get("index"), int) else float("inf"),
            item.get("label") or "",
        )
    )

    return per_core


def _collect_memory_metrics() -> dict[str, object] | None:
    """Return memory utilisation metrics if available."""

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:  # pragma: no cover - defensive guard for non-Linux platforms
        return None

    total_kib = available_kib = None
    for line in lines:
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    total_kib = int(parts[1])
                except ValueError:
                    pass
        elif line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    available_kib = int(parts[1])
                except ValueError:
                    pass
        if total_kib is not None and available_kib is not None:
            break

    if total_kib is None or total_kib <= 0:
        return None

    metrics: dict[str, object] = {"total_bytes": total_kib * 1024}

    if available_kib is not None:
        available_bytes = max(0, available_kib * 1024)
        metrics["available_bytes"] = available_bytes
        used_bytes = max(0, metrics["total_bytes"] - available_bytes)
        metrics["used_bytes"] = used_bytes
        if metrics["total_bytes"]:
            metrics["used_percent"] = _clamp_percentage(
                (used_bytes / metrics["total_bytes"]) * 100.0
            )

    return metrics


def _format_bytes(value: int) -> str:
    """Return *value* in bytes as a human-readable string."""

    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    size = float(value)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    if size >= 100:
        formatted = f"{size:.0f}"
    elif size >= 10:
        formatted = f"{size:.1f}"
    else:
        formatted = f"{size:.2f}"
    return f"{formatted} {units[unit_index]}"


def collect_system_metrics() -> dict[str, object]:
    """Return a summary of system-level resource usage."""

    payload: dict[str, object] = {}

    cpu_metrics = _collect_cpu_metrics()
    if cpu_metrics:
        payload["cpu"] = cpu_metrics

    memory_metrics = _collect_memory_metrics()
    if memory_metrics:
        payload["memory"] = memory_metrics

    return payload


def collect_diagnostics() -> dict[str, object]:
    """Collect diagnostics payload used by both the CLI and API."""

    hints = diagnose_camera_conflicts()
    picamera_stack = diagnose_picamera_stack()
    webrtc_stack = diagnose_webrtc_stack()
    system_metrics = collect_system_metrics()
    payload: dict[str, object] = {
        "version": APP_VERSION,
        "camera_conflicts": hints,
        "picamera": picamera_stack,
        "webrtc": webrtc_stack,
    }
    if system_metrics:
        payload["system"] = system_metrics
    return payload


def run(argv: Sequence[str] | None = None) -> int:
    """Execute the diagnostics CLI with *argv* arguments."""

    parser = build_parser()
    args = parser.parse_args(argv)

    payload = collect_diagnostics()
    camera_conflicts = payload.get("camera_conflicts", [])
    picamera_stack = payload.get("picamera", {})
    webrtc_stack = payload.get("webrtc", {})
    system_metrics = payload.get("system", {})

    if not isinstance(camera_conflicts, list):
        camera_conflicts = list(camera_conflicts) if camera_conflicts else []

    if not isinstance(picamera_stack, dict):
        picamera_stack = {}
    if not isinstance(webrtc_stack, dict):
        webrtc_stack = {}
    if not isinstance(system_metrics, dict):
        system_metrics = {}

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

    if system_metrics:
        print("System resource usage:")
        cpu_metrics = system_metrics.get("cpu")
        if isinstance(cpu_metrics, dict) and cpu_metrics:
            cpu_parts: list[str] = []
            usage_percent = cpu_metrics.get("usage_percent")
            if isinstance(usage_percent, (int, float)) and usage_percent >= 0:
                cpu_parts.append(f"{usage_percent:.1f}% (1m avg)")
            load = cpu_metrics.get("load")
            if isinstance(load, dict):
                load_1m = load.get("1m")
                if isinstance(load_1m, (int, float)):
                    cpu_parts.append(f"load 1m {load_1m:.2f}")
            cpu_count = cpu_metrics.get("count")
            if isinstance(cpu_count, int) and cpu_count > 0:
                cpu_parts.append(f"{cpu_count} cores")
            cpu_summary = ", ".join(cpu_parts) if cpu_parts else "data unavailable"
            print(f" - CPU: {cpu_summary}")
            per_core_metrics = cpu_metrics.get("per_core")
            if isinstance(per_core_metrics, list):
                per_core_parts: list[str] = []
                for entry in per_core_metrics:
                    if not isinstance(entry, dict):
                        continue
                    usage = entry.get("usage_percent")
                    if not isinstance(usage, (int, float)):
                        continue
                    label: str | None = None
                    core_index = entry.get("index")
                    if isinstance(core_index, int) and core_index >= 0:
                        label = f"core {core_index}"
                    else:
                        raw_label = entry.get("label")
                        if isinstance(raw_label, str) and raw_label:
                            label = raw_label
                    if label:
                        per_core_parts.append(f"{label} {usage:.1f}%")
                    else:
                        per_core_parts.append(f"{usage:.1f}%")
                if per_core_parts:
                    joined = ", ".join(per_core_parts)
                    print(f"   Per-core usage: {joined}")
        memory_metrics = system_metrics.get("memory")
        if isinstance(memory_metrics, dict) and memory_metrics:
            memory_parts: list[str] = []
            used_percent = memory_metrics.get("used_percent")
            if isinstance(used_percent, (int, float)) and used_percent >= 0:
                memory_parts.append(f"{used_percent:.1f}% used")
            used_bytes = memory_metrics.get("used_bytes")
            total_bytes = memory_metrics.get("total_bytes")
            if isinstance(used_bytes, int) and isinstance(total_bytes, int):
                memory_parts.append(
                    f"{_format_bytes(used_bytes)} / {_format_bytes(total_bytes)}"
                )
            elif isinstance(total_bytes, int):
                memory_parts.append(f"total {_format_bytes(total_bytes)}")
            memory_summary = ", ".join(memory_parts) if memory_parts else "data unavailable"
            print(f" - Memory: {memory_summary}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by `python -m rev_cam.diagnostics`."""

    return run(argv)


__all__ = [
    "build_parser",
    "diagnose_picamera_stack",
    "diagnose_webrtc_stack",
    "collect_system_metrics",
    "collect_diagnostics",
    "run",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - module behaviour
    sys.exit(main())
