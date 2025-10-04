# Action Plan for Identified Issues

The repository highlights several latency-related issues that would benefit from focused follow-up work. The sections below translate each reported problem into actionable next steps so contributors can prioritise improvements and track progress.

## 1. Distance sensor sampling blocks the frame pipeline
- **Issue summary:** Distance reads currently happen synchronously in the distance overlay, so any I²C delays stall frame processing and introduce jitter.【F:docs/latency_improvement_tasks.md†L4-L12】
- **Recommended actions:**
  - Move VL53L1X sampling into a dedicated background worker (thread or async task) that maintains the latest reading.
  - Update the overlay to consume cached readings and ensure the worker exposes health metrics so diagnostics can detect failures.
  - Benchmark frame cadence before and after to confirm latency improvements and document the results.

## 2. MJPEG transforms run on the asyncio event loop
- **Issue summary:** Frame processing executes inline on the asyncio loop, competing with connection management and increasing latency.【F:docs/latency_improvement_tasks.md†L14-L22】
- **Recommended actions:**
  - Offload the entire MJPEG processing path to a worker executor shared with JPEG encoding or an equivalent background task.
  - Mirror the WebRTC pipeline's `asyncio.to_thread(...)` approach so the event loop remains focused on I/O.
  - Capture frame interval metrics pre/post change to ensure the refactor delivers measurable jitter reduction.

## 3. Configuration reads cause lock contention
- **Issue summary:** Per-frame overlays repeatedly hit `ConfigManager` locks, leading to contention and micro-stalls at higher frame rates.【F:docs/latency_improvement_tasks.md†L24-L31】
- **Recommended actions:**
  - Introduce lock-free snapshots or publish/subscribe hooks that refresh cached values only when settings change.
  - Audit hotspots for redundant configuration queries and consolidate lookups where possible.
  - Validate the approach with stress tests that simulate rapid config changes alongside streaming workloads.

## 4. Text overlays allocate new buffers every frame
- **Issue summary:** The `_blend_text` helper allocates fresh NumPy buffers and recalculates layout metrics for each frame, inflating CPU time and GC pressure.【F:docs/latency_improvement_tasks.md†L33-L40】
- **Recommended actions:**
  - Precompute glyph rasters for the static strings used in overlays and reuse scratch buffers for blending.
  - Replace per-frame float conversions with in-place uint8 operations to shrink the hot path.
  - Add profiling benchmarks that compare memory allocations and CPU usage before and after optimisation.

## 5. Reversing-aid overlay rasterises lines in Python loops
- **Issue summary:** The current implementation stamps pixels point-by-point, making high-resolution frames expensive to render.【F:docs/latency_improvement_tasks.md†L42-L47】
- **Recommended actions:**
  - Replace manual Bresenham loops with vectorised NumPy operations or leverage OpenCV drawing primitives when available.
  - Provide a fallback implementation for environments without OpenCV, ensuring feature parity across builds.
  - Measure render times across multiple resolutions to verify the new approach scales better.

---
These action items give contributors a clear starting point for tackling the most impactful latency issues while leaving room for experimentation with specific libraries and implementation details.
