# Latency Improvement Opportunities

The review below highlights hotspots in the current pipeline that can introduce avoidable latency or jitter at runtime. Each item includes suggested follow-up work to address the issue.

## 1. Offload distance sensor reads from the frame pipeline
*Problem*: The distance overlay calls `DistanceMonitor.read()` synchronously for every frame (`create_distance_overlay`). That routine acquires locks and can block on I2C calls before returning a reading, which means frame processing stalls whenever the sensor is slow or transiently unavailable.【F:src/rev_cam/distance.py†L604-L645】【F:src/rev_cam/distance.py†L505-L555】

*Task*: Move the VL53L1X sampling loop into its own async task or thread that continually refreshes a cached reading. The overlay should consume the latest cached value without performing I/O on the render path. This decouples slow sensor hardware from the camera frame cadence and reduces latency spikes. Measure end-to-end latency before/after to verify improvement.

## 2. Run MJPEG pipeline transforms off the event loop
*Problem*: The MJPEG producer executes `FramePipeline.process()` inline on the event loop and only offloads JPEG encoding to a worker thread. Overlay rendering performs multiple CPU-heavy operations (e.g., distance text blending and reversing-aid rasterisation), so running it inline competes with connection handling and increases frame times.【F:src/rev_cam/streaming.py†L229-L258】【F:src/rev_cam/pipeline.py†L22-L48】

*Task*: Shift the entire frame processing step into the same worker thread used for JPEG encoding (or a dedicated executor) so the asyncio loop just orchestrates I/O. Alternatively, reuse the WebRTC pathway’s approach (`asyncio.to_thread(self._manager.pipeline.process, frame)`) for MJPEG. Profile frame interval jitter before and after to confirm the reduced loop blocking.

## 3. Cache frequently polled configuration values
*Problem*: Per-frame overlays repeatedly call config accessors that acquire `ConfigManager`’s lock (e.g., orientation in `FramePipeline.process`, zone thresholds in the distance overlay). At 20–30 FPS this adds constant lock churn and can serialize unrelated config writes, causing micro-stalls.【F:src/rev_cam/pipeline.py†L45-L48】【F:src/rev_cam/config.py†L620-L768】【F:src/rev_cam/distance.py†L622-L639】

*Task*: Introduce lightweight caching or snapshot objects that are updated only when settings change (e.g., via publish/subscribe callbacks or atomic dataclasses). The frame pipeline and overlays should read from lock-free copies, while writes still go through the manager. Benchmark the frame loop with and without caching to quantify the improvement.

## 4. Reduce per-frame allocation in text overlays
*Problem*: `_blend_text` allocates new NumPy buffers, converts data to float32, and recomputes layout metrics for every rendered string each frame. These allocations and conversions show up in profiles as avoidable CPU time and pressure the allocator.【F:src/rev_cam/distance.py†L677-L762】

*Task*: Precompute glyph bitmaps and layout metrics for the fixed set of strings, reuse scratch buffers, and perform blending in-place with uint8 arithmetic. This will shrink the per-frame workload and make the overlay more deterministic.

## 5. Vectorise reversing-aid rasterisation
*Problem*: The reversing-aid overlay stamps each line point-by-point in Python, repeatedly calling `_stamp_disc` with nested loops. On high-resolution frames this tight Python loop becomes a noticeable CPU cost and lengthens frame processing time.【F:src/rev_cam/reversing_aids.py†L23-L122】

*Task*: Replace the manual Bresenham implementation with NumPy-based rasterisation (e.g., drawing into a mask array and blending once) or leverage OpenCV primitives when available. This keeps the overlay flexible while dramatically reducing the Python overhead per frame.

