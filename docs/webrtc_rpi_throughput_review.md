# WebRTC Raspberry Pi Throughput Review

This document captures the current state of RevCam's WebRTC pipeline on the Raspberry Pi Zero 2 W and tracks the follow-up work required to raise sustained throughput. It complements the open checklist in [`webrtc_rpi_throughput_task_list.md`](./webrtc_rpi_throughput_task_list.md) by documenting what has been verified and which gaps remain.

## 1. Baseline assessment

| Task | Status | Findings |
| --- | --- | --- |
| Inventory the active Raspberry Pi hardware | ✅ | RevCam explicitly targets the Raspberry Pi Zero 2 W running Raspberry Pi OS Bookworm, giving a quad-core Cortex-A53 CPU with the VideoCore VI multimedia pipeline and 512 MB RAM to budget for capture and encode workloads.【F:README.md†L1-L6】 |
| Document the existing WebRTC pipeline architecture | ✅ | Frames are pulled from the configured `BaseCamera`, processed through `FramePipeline` on a worker thread, converted to RGB24, and injected into an `aiortc` `MediaStreamTrack` for SDP negotiation by `WebRTCManager`. The track enforces the configured frame cadence, while the manager owns the peer connection lifecycle.【F:src/rev_cam/streaming.py†L291-L355】【F:src/rev_cam/streaming.py†L392-L458】 |
| Capture current throughput metrics | 🚧 | No automated bitrate or frame pacing instrumentation exists yet. Browser-side stats and `RTCRtpSender.getStats()` should be captured during representative scenes alongside CPU and temperature samples from `diagnostics.collect_cpu_metrics()` to establish the baseline envelope.【F:src/rev_cam/diagnostics.py†L180-L208】 |
| Record encoder settings | ✅ | The WebRTC path does not yet expose an H.264 encoder configuration. Frames are delivered to `aiortc` as raw RGB, and codec, profile, level, bitrate, and GOP decisions are left to the default Safari ↔︎ `aiortc` negotiation. Effective defaults must be captured once instrumentation is in place to guard against regression. |

## 2. Hardware vs. software encoding analysis

| Task | Status | Findings |
| --- | --- | --- |
| Enumerate available hardware encoders | ✅ | Pi OS exposes the VideoCore VI H.264 encoder through V4L2 M2M (`h264_v4l2m2m`) and legacy OMX (`h264_omx`). These codecs are already probed for recordings and can be surfaced for WebRTC once the pipeline produces YUV buffers.【F:src/rev_cam/recording.py†L32-L66】 |
| Profile CPU utilisation, power, thermals for software encode | 🚧 | `diagnostics.collect_cpu_metrics()` reports per-core load and SoC temperature, but WebRTC sessions currently lack hooks to sample these metrics under load. Add periodic polling tied to the peer connection lifecycle so trials at 2/4/8 Mbps can be compared.【F:src/rev_cam/diagnostics.py†L180-L208】 |
| Compare hardware vs. software encoders | ⬜ | Hardware encoding is not wired into the WebRTC path; throughput A/B testing requires a new encoder abstraction capable of supplying Annex B NAL units to the RTP packetiser. |
| Identify throughput failure modes | 🚧 | Thermal and backpressure failures are not yet observable. Planned metrics (CPU load, temperature, dropped frame counters) combined with aiortc stats will highlight throttling or queuing hotspots once added. |

## 3. Encoder configuration & bitrate strategy (H.264)

| Task | Status | Findings |
| --- | --- | --- |
| Evaluate presets and rate control modes | ⬜ | The current pipeline leaves rate control entirely to aiortc’s defaults. Encoder integration must expose CBR/VBR selection with scene-change responsiveness validated once instrumentation exists. |
| Tune H.264 profile/level | ⬜ | No profile or level is forced today; the target should align with the Pi hardware encoder (Baseline/Constrained Baseline Level 4.0 or lower for iOS compatibility). |
| Test B-frames vs. P-only | ⬜ | aiortc defaults are unknown on Raspberry Pi; once a dedicated encoder backend is added, toggling B-frames must be part of throughput validation. |
| Validate rate control responsiveness | ⬜ | Requires bitrate telemetry from the congestion controller plus encoder buffer statistics; pending tooling work in §8. |

## 4. GOP structure (~2 s)

| Task | Status | Findings |
| --- | --- | --- |
| Confirm current GOP length | ⬜ | With no encoder hooks, GOP length is implicit. Add encoder configuration surfaces that default to a 2 s keyframe interval (e.g., 40 frames at 20 fps) and log the negotiated IDR cadence. |
| Assess keyframe spikes | ⬜ | Blocked on encoder telemetry; capture RTP packet sizes around keyframes once instrumentation is in place. |
| Explore IDR spacing strategies | ⬜ | Requires the encoder work above. |
| Verify decoder tolerance | 🚧 | Safari clients have been stable with the current defaults, but explicit validation with the forthcoming encoder options is outstanding. |

## 5. Event-loop backpressure & pipeline flow control

| Task | Status | Findings |
| --- | --- | --- |
| Trace backpressure propagation | ✅ | `PipelineVideoTrack.recv()` awaits camera frames, processes them on a worker via `asyncio.to_thread`, and enforces frame pacing with `asyncio.sleep`. There is no explicit queue beyond the aiortc sender buffers, so backpressure currently surfaces as slower `recv()` completions.【F:src/rev_cam/streaming.py†L319-L354】 |
| Instrument loop latency and buffer occupancy | 🚧 | Need to time `camera.get_frame()` plus pipeline processing and log when `remaining` in `recv()` turns negative to detect overruns. |
| Implement adaptive backpressure | ⬜ | Future work should drop frames or lower fps when processing time exceeds the frame interval instead of blocking the sender. |
| Validate interplay with WebRTC pacing | ⬜ | Blocked until adaptive controls and pacing metrics are available. |

## 6. Packetisation & MTU considerations

| Task | Status | Findings |
| --- | --- | --- |
| Audit RTP packetisation | 🚧 | aiortc currently owns packetisation. Once H.264 NAL units are supplied directly, ensure the sender uses STAP-A aggregation and FU-A fragmentation aligned with the chosen MTU. |
| Measure network MTU | ⬜ | Add scripted `ping -M do` sweeps in the test harness to capture MTU for Ethernet vs. Wi‑Fi deployments. |
| Evaluate STUN/TURN overhead | ⬜ | Include DTLS + SRTP expansion in MTU calculations when documenting the effective payload budget. |
| Test packet loss sensitivity | ⬜ | Needs network emulation once MTU tuning is complete. |

## 7. WebRTC congestion control tuning

| Task | Status | Findings |
| --- | --- | --- |
| Document libwebrtc stack | ✅ | RevCam embeds aiortc for signalling and media transport; `scripts/install_prereqs.sh` ensures libsrtp2, libopus, and libvpx are installed to unlock WebRTC media support.【F:scripts/install_prereqs.sh†L46-L97】 |
| Collect transport feedback stats | ⬜ | Pending addition of `RTCRtpSender.getStats()` polling and log export. |
| Adjust encoder targets from congestion feedback | ⬜ | Blocked on encoder integration; design should feed aiortc bitrate updates into the encoder’s rate control. |
| Validate bandwidth estimation behaviour | ⬜ | Requires the telemetry plumbing above. |

## 8. Tooling & automation

| Task | Status | Findings |
| --- | --- | --- |
| Set up repeatable test harness | 🚧 | Existing FastAPI endpoints allow scripted session startup, but no automated harness exists. Add pytest-style fixtures or standalone scripts that open a headless WebRTC session and drive tc/netem for impairments. |
| Automate metric collection | 🚧 | Diagnostics already gather CPU stats; extend them to push structured logs (JSONL or Influx line protocol) with bitrate, RTT, and loss once stats polling is implemented.【F:src/rev_cam/diagnostics.py†L180-L208】 |
| Define acceptance thresholds | ⬜ | To be finalised after baseline metrics are captured. |

## 9. Documentation & knowledge sharing

| Task | Status | Findings |
| --- | --- | --- |
| Summarise hardware/software trade-offs | 🚧 | This document captures the current architecture and gaps; complete once encoder benchmarking data is available. |
| Provide operational runbooks | ⬜ | Pending encoder configuration surfaces and congestion-control tuning guidance. |
| Highlight remaining risks | ✅ | Key risks include lack of encoder configurability, absent throughput telemetry, and unverified congestion-control behaviour noted above. |

## Next steps

1. Instrument aiortc stats and CPU/thermal sampling during WebRTC sessions.
2. Introduce an H.264 encoder abstraction that can switch between software (libx264) and hardware (V4L2 M2M) backends.
3. Build a repeatable network-emulation harness to exercise congestion-control and MTU scenarios.
4. Iterate through Sections 3–8 once instrumentation exists, updating this document and the task list with measured outcomes.
