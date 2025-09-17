# RevCam

RevCam is a low-latency reversing camera stack targeted at the Raspberry Pi Zero 2 W (Bookworm).
It streams a live camera feed to iOS devices through WebRTC and exposes a settings panel for
configuring image orientation. The processing pipeline is modular so that future driver-assistance
overlays can be injected on the server without major changes.

## Features

- Fast WebRTC video delivery optimised for mobile Safari (iPhone/iPad).
- Camera orientation controls (rotation and horizontal/vertical flips).
- Modular frame processing pipeline ready for future overlays (e.g. guidelines).
- REST API for orientation control and WebRTC signalling.

## Project layout

```
src/
  rev_cam/
    __init__.py
    app.py                # FastAPI application and routes
    camera.py             # Camera source abstractions
    config.py             # Orientation persistence and validation
    pipeline.py           # Frame processing pipeline (orientation + overlays)
    webrtc.py             # WebRTC track implementation
  rev_cam/static/
    index.html            # Viewer client
    settings.html         # Settings UI
```

## Getting started

The project targets Python 3.11+. On development machines install the dependencies with

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Camera support is pluggable:

- **Raspberry Pi camera** – Install the system packages provided by Raspberry Pi
  OS instead of pulling `picamera2` from PyPI. This avoids the `python-prctl`
  build dependency on `libcap` headers. On a Pi run:

  ```bash
  sudo apt install python3-picamera2 python3-prctl
  ```

  The packaged build already bundles the compiled dependencies required by
  Picamera2, so no additional Python packages are needed.
- **OpenCV USB camera** – Install `opencv-python` manually and set
  `REVCAM_CAMERA=opencv`.
- **Synthetic frames** – For development without camera hardware set
  `REVCAM_CAMERA=synthetic` (default when Picamera2 is unavailable).

Run the server with

```bash
uvicorn rev_cam.app:create_app --factory --host 0.0.0.0 --port 8000
```

Then open `http://<pi-address>:8000` on the iOS device to view the stream. Access the
settings panel at `/settings` to adjust the camera orientation. Changes are persisted
and applied immediately to the outgoing WebRTC stream.

## Testing

```
pytest
```

Unit tests cover the frame pipeline and configuration management so that critical
behaviour remains reliable.

## Future work

- Integrate computer-vision overlays (parking guidelines, obstacle detection).
- Replace the default software encoder with hardware acceleration.
- Add authentication for configuration endpoints.

