"""RevCam package exposing factory helpers for the reversing camera server."""

from typing import Any


def create_app(*args: Any, **kwargs: Any):
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
