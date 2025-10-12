# WebRTC Raspberry Pi Throughput Review Task List

## 1. Baseline Assessment
- [ ] Inventory the current Raspberry Pi model(s), firmware revision, and GPU/CPU capabilities.
- [ ] Document existing WebRTC pipeline architecture (capture -> encode -> packetize -> transport -> decode).
- [ ] Capture current end-to-end throughput metrics (average bitrate, peak bitrate, frame rate, packet loss) under typical workloads.
- [ ] Record current encoder settings: codec (H.264), profile, level, bitrate target, GOP length, and latency requirements.

## 2. Hardware vs. Software Encoding Analysis
- [ ] Enumerate available hardware encoders (e.g., VideoCore VI, OMX, V4L2 M2M) and their driver support on the Pi.
- [ ] Profile CPU utilization, power draw, and thermal headroom when using software encoding (x264 or libavcodec) across multiple bitrates.
- [ ] Measure throughput and latency differences between hardware-accelerated and software encoders under identical scene complexity.
- [ ] Identify failure modes (thermal throttling, dropped frames) that may cap throughput.

## 3. Encoder Configuration & Bitrate Strategy (H.264)
- [ ] Evaluate encoder presets and rate control modes (CBR, VBR, Constrained VBR) for target bitrates (e.g., 2, 4, 8 Mbps).
- [ ] Tune H.264 profile/level settings to match Pi hardware capabilities and receiving client constraints.
- [ ] Test effect of B-frames vs. P-only streams on throughput and latency.
- [ ] Validate rate control responsiveness to scene changes and identify buffer sizes that prevent bitrate overshoot.

## 4. GOP Structure (≈ 2s)
- [ ] Confirm current GOP length (keyframe interval) and alignment with 2-second target at multiple frame rates.
- [ ] Assess keyframe size spikes and their effect on bandwidth bursts and congestion controller behavior.
- [ ] Explore use of IDR frame spacing vs. CRA/Recovery frames to smooth bandwidth usage.
- [ ] Verify downstream decoders tolerate the selected GOP structure without increasing latency.

## 5. Event-Loop Backpressure & Pipeline Flow Control
- [ ] Trace capture-to-encode-to-send pipeline to identify where backpressure signals propagate (e.g., async queues, GStreamer appsink).
- [ ] Instrument event loop latency and buffer occupancy to detect congestion in software threads.
- [ ] Implement or refine backpressure mechanisms (dropping frames, dynamic quality scaling) when encoder or network cannot keep up.
- [ ] Validate that backpressure interacts correctly with WebRTC’s internal pacing to avoid bursty transmissions.

## 6. Packetization & MTU Considerations
- [ ] Audit RTP packetization settings (NALU aggregation, fragmentation units) used for H.264 payloads.
- [ ] Measure actual MTU on target networks (Ethernet, Wi-Fi) and adjust max payload size to avoid IP fragmentation.
- [ ] Evaluate STUN/TURN overhead and DTLS/SRTP expansion to compute effective payload size limits.
- [ ] Test packet loss sensitivity when large NAL units are fragmented across multiple packets.

## 7. WebRTC Congestion Control Tuning
- [ ] Document current WebRTC stack (libwebrtc version, customizations) and its default congestion control algorithm (GCC, BBR, SCReaM).
- [ ] Collect transport feedback stats (RTT, packet loss, send-side bitrate) under controlled network impairments.
- [ ] Adjust encoder target bitrate and pacing in response to congestion controller feedback to maintain throughput stability.
- [ ] Validate bandwidth estimation behavior with keyframe bursts and rapid bitrate changes.

## 8. Tooling & Automation
- [ ] Set up repeatable test harness (network emulation, scripted WebRTC sessions) to compare changes.
- [ ] Automate metric collection (influxDB/Grafana, WebRTC internals dumps, custom logs) for throughput analysis.
- [ ] Define acceptance thresholds for throughput, latency, and stability across scenarios.

## 9. Documentation & Knowledge Sharing
- [ ] Summarize findings for hardware vs. software encoding trade-offs and recommended encoder settings.
- [ ] Provide runbooks for adjusting GOP, bitrate, and congestion control parameters in production.
- [ ] Highlight remaining risks and propose future experiments (e.g., AV1, SVC, multi-stream scenarios).

