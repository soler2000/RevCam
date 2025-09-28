# Surveillance Mode Functional Specification

## 1. Goals & Non-Goals
- **Goal:** Provide continuous and lightweight monitoring that records motion-triggered clips with pre-roll, gallery view, and bulk export/delete.
- **Non-Goals:** No overlays, no simultaneous use with Reversing mode, and no analytics beyond motion detection (e.g., recognition or object tracking).

## 2. Hard Separation From Reversing Mode
- Surveillance and Reversing modes are mutually exclusive; only one may access the camera at a time.
- Introduce a singleton **Mode Manager** exposing `POST /api/mode { "mode": "reversing" | "surveillance" }`.
- Enforce a camera device lock (e.g., `/run/revcam/camera.lock`). Both modes refuse to start if the lock is held.
- Processes/services:
  - `revcam-reversing.service` (existing).
  - `revcam-surveillance.service` (new).
  - `revcam-mode.service` (broker responsible for safe mode switching, or embedded in the app).
- Resources: No overlays. Introduce a `LedController` that ensures exclusive LED ownership and throws if the current mode does not own it.

## 3. Capture & Motion Pipeline
- **Camera stack:** Picamera2 with libcamera H.264 encoder.
- **Recording format:** MP4 (H.264/AAC or silent). Optional HEVC if hardware supports it.
- **Pre-roll:** Circular RAM buffer (e.g., Picamera2 `CircularOutput`) with configurable seconds.
- **Motion detection:**
  - Downscaled grayscale frames (~160×90) at 5–10 FPS.
  - Frame differencing with adaptive background/running average.
  - Tunables: sensitivity (pixel change threshold), minimum changed area percentage, minimum motion duration (debounce).
  - Optional privacy masks (rectangles) to ignore regions.
- **Clip boundaries:**
  - Start when motion sustains for `min_motion_duration_ms`; dump pre-roll and start recording.
  - Stop after `post_motion_gap_ms` without motion; cap clips at `max_clip_length_s` and roll into new clips if motion persists.
- **LED behavior:** Configurable flash or steady state upon motion trigger; no on-video overlay.

## 4. Files, Storage & Retention
- **Directory layout:** `/var/lib/revcam/surveillance/YYYY/MM/DD/`.
- **File naming:** `clip-<UTC_ISO8601>-<uuid-short>.mp4` and matching thumbnails `clip-<same>.jpg`.
- **Metadata DB:** SQLite at `/var/lib/revcam/surveillance/index.db` with table `clips(id, path, start_ts, end_ts, duration_s, size_bytes, has_audio, thumb_path, motion_score, settings_hash)`.
- **Retention policy:** Configurable maximum total size (GB) and/or days. Purge oldest clips first, never during active recording.

## 5. Settings (UI & Config)
### UI Controls
- **Recording:**
  - Resolution: 640×360, 1280×720, 1920×1080 (filtered by device capability).
  - Framerate: 10–30 FPS.
  - Encoding: H.264 (default) or HEVC (if supported).
  - Clip max length: 10–300 seconds.
  - Pre-roll: 0–10 seconds (default 3).
  - Post-motion gap: 1–15 seconds (default 4).
  - Audio toggle (if microphone available).
- **Motion:**
  - Sensitivity (low/medium/high or numeric threshold).
  - Minimum changed area percentage.
  - Minimum motion duration (ms).
  - Detection FPS (5–10).
  - Privacy masks (add/edit rectangles).
- **Actions on motion:**
  - Record toggle.
  - LED mode: off / steady / flash (with rate).
  - Webhook: URL, method, payload template (optional).
  - Notifications: reserved for future.
- **Storage:**
  - Max retention days and/or max size (GB).
  - Export path (default temporary ZIP).
- Provide Apply/Save buttons with validation.

## 6. New Pages & UX
### Surveillance Page (new main navigation entry)
- **Player pane:** HTML5 `<video>` element showing selected clip with filename, timestamps, duration, size, play/pause/seek controls, and per-clip download button.
- **Thumbnail gallery:** Grid of clip cards (thumbnail, timestamp, duration, size) with filters (date range, motion score, resolution, size), pagination or infinite scroll, and multi-select checkboxes.
- **Bulk actions:** Export to ZIP and Delete for selected clips.

### Settings → Surveillance Tab
- Hosts the controls enumerated above.

## 7. API Endpoints
- **Mode control:**
  - `GET /api/mode` → `{ "mode": "reversing" | "surveillance" }`.
  - `POST /api/mode` with `{ "mode": "surveillance" }` to safely switch modes.
- **Settings:**
  - `GET /api/surv/settings`.
  - `PUT /api/surv/settings` to validate, persist, and reload.
- **Clips:**
  - `GET /api/surv/clips?from=ISO&to=ISO&page=1&page_size=50&sort=-start_ts` returning clip summaries.
  - `GET /api/surv/clips/{id}` for metadata.
  - `GET /api/surv/clips/{id}/stream` for MP4 byte-range streaming.
  - `GET /api/surv/clips/{id}/thumb` for JPEG thumbnails.
  - `POST /api/surv/clips/export` with `{ "ids": [...] }` returning `{ "zip_url": "/download/export-<uuid>.zip" }`.
  - `DELETE /api/surv/clips` with `{ "ids": [...] }` for bulk deletion.
- **Actions:**
  - `POST /api/surv/test-motion` to simulate motion for action validation.
  - `POST /api/surv/purge` to immediately apply retention policy.

## 8. Recorder State Machine
- **Idle:** Pre-roll buffer active, not writing to disk.
- **Arming:** Motion detected, waiting for minimum duration.
- **Recording:** MP4 capture active; LED per configuration.
- **Stopping:** No motion for post gap or max clip length reached.

Transitions follow motion debouncing and timer conditions.

## 9. Performance Targets
- Target Raspberry Pi Zero 2 W: 720p @ 15 FPS H.264 recording, motion detection on 160×90 @ 6 FPS consuming low CPU.
- Disk I/O: sequential writes ≤ 5–8 MB/s during events.
- Thumbnails generated from first keyframe (or mid-clip) using `ffmpeg -ss` at low priority.

## 10. Systemd & Start/Stop Safety
- **`revcam-surveillance.service`:**
  - `After=network-online.target`.
  - `ExecStart=revcam --mode surveillance`.
  - Pre-start waits for camera lock and fails fast if busy.
- **`revcam-reversing.service`:** unchanged but respects camera lock.
- Mode switch helper stops the active service, waits for lock release, then starts the requested mode.

## 11. LED Control
- Reuse GPIO18 with an exclusive LED controller per mode.
- Flash patterns managed by an async task and halted on mode exit.

## 12. Errors & Edge Cases
- Out of space: stop recording, surface UI banner, prompt export/delete.
- Camera busy (lock held): API returns HTTP 423 Locked with explanation.
- Power loss corruption: attempt MP4 `moov` atom repair on boot and mark clips as "recovering".

## 13. Security
- `/api/surv/*` uses the same authentication as the main app.
- Export ZIPs saved to `/var/lib/revcam/exports/` with short-lived signed URLs or requiring logged-in session.

## 14. Testing Checklist
- **Unit:** motion thresholding, pre/post timers, retention purge, DB CRUD.
- **Integration:** mode switching, camera lock enforcement, LED flashing, webhook firing.
- **End-to-end:** pre-roll recording, gallery rendering, multi-select export/delete, settings persistence.

## 15. Milestone Breakdown
1. Core engine (pre-roll, motion detection, recording, finalize).
2. Database and thumbnail generation.
3. API implementation (list/stream/export/delete).
4. UI pages (player, gallery, settings tab).
5. Mode manager and systemd integration.
6. Retention handling and error surfaces.
7. Documentation and examples.
