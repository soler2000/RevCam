"""Measure overlay latency impact of background VL53L1X sampling."""

from __future__ import annotations

import pathlib
import statistics
import sys
import time
from typing import Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rev_cam.distance import DistanceMonitor, DistanceZones, create_distance_overlay


class _SlowSensor:
    def __init__(self, delay: float, value: float = 1500.0) -> None:
        self._delay = float(delay)
        self._value = float(value)

    @property
    def distance(self) -> float:
        time.sleep(self._delay)
        return self._value


def _measure(overlay: Callable[[np.ndarray], np.ndarray], iterations: int = 20) -> float:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        overlay(frame)
        timings.append(time.perf_counter() - start)
    return statistics.mean(timings)


def main() -> None:
    delay = 0.05
    iterations = 20

    baseline_sensor = _SlowSensor(delay)
    baseline_monitor = DistanceMonitor(
        sensor_factory=lambda: baseline_sensor,
        update_interval=0.0,
        auto_start=False,
    )

    def _synchronous_overlay(frame: np.ndarray) -> np.ndarray:
        baseline_monitor.refresh()
        return frame

    baseline_latency = _measure(_synchronous_overlay, iterations)

    async_sensor = _SlowSensor(delay)
    async_monitor = DistanceMonitor(
        sensor_factory=lambda: async_sensor,
        update_interval=delay,
        auto_start=False,
    )
    zones = DistanceZones(2.0, 1.0, 0.5)
    overlay = create_distance_overlay(async_monitor, lambda: zones)

    async_monitor.refresh()
    time.sleep(delay)

    async_latency = _measure(overlay, iterations)

    baseline_ms = baseline_latency * 1_000
    async_ms = async_latency * 1_000

    print(f"Synchronous overlay average: {baseline_ms:.1f} ms over {iterations} frames")
    print(f"Background sampling average: {async_ms:.1f} ms over {iterations} frames")
    print(f"Improvement: {baseline_ms - async_ms:.1f} ms per frame")

    baseline_monitor.close()
    async_monitor.close()


if __name__ == "__main__":
    main()
