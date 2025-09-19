"""Command-line helpers for RevCam diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .camera import diagnose_camera_conflicts
from .version import APP_VERSION


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


def run(argv: Sequence[str] | None = None) -> int:
    """Execute the diagnostics CLI with *argv* arguments."""

    parser = build_parser()
    args = parser.parse_args(argv)

    hints = diagnose_camera_conflicts()
    payload = {
        "version": APP_VERSION,
        "camera_conflicts": hints,
    }

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
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by `python -m rev_cam.diagnostics`."""

    return run(argv)


__all__ = ["build_parser", "run", "main"]


if __name__ == "__main__":  # pragma: no cover - module behaviour
    import sys

    sys.exit(main())
