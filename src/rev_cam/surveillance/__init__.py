"""Surveillance mode support for RevCam."""

from .manager import ClipFilters, SurveillanceManager
from .mode import ModeController, ModeManager, ModeSwitchError
from .runtime import MotionRecorder, SurveillanceRuntime
from .settings import LedBehaviour, PrivacyMask, SurveillanceSettings

__all__ = [
    "ClipFilters",
    "LedBehaviour",
    "ModeController",
    "MotionRecorder",
    "ModeManager",
    "ModeSwitchError",
    "PrivacyMask",
    "SurveillanceManager",
    "SurveillanceSettings",
    "SurveillanceRuntime",
]
