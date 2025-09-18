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

RevCam targets Python 3.11+.

### Raspberry Pi installation

The Raspberry Pi wheels for Picamera2 and its native dependencies are shipped
through Raspberry Pi OS. Using them is much faster and more reliable than
building everything from PyPI. Follow these steps on the Pi:

1. Install the packaged Picamera2 stack and its helpers:

   ```bash
   sudo apt update
   sudo apt install python3-picamera2 python3-prctl
   ```

2. Create a virtual environment that can see the system packages you just
   installed (this prevents `pip` from trying to rebuild Picamera2 and PiDNG):

   ```bash
   python3 -m venv --system-site-packages .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

3. Install RevCam from the source tree. On Pi hardware some wheels still need to
   be built locally (`aiortc`, `pylibsrtp`, `av`), so this step can take several
   minutes. Seeing a long list ending with `Successfully installed ...` means
   the step finished successfully:

   ```bash
   pip install --prefer-binary --extra-index-url https://www.piwheels.org/simple -e .
   ```

   The `--prefer-binary` flag asks `pip` to fetch pre-built wheels when
   available, and the PiWheels index provides ARM builds for most dependencies.

### Development machine installation

For local development on non-Pi machines a regular virtual environment is
sufficient:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Camera back-ends

Camera support is pluggable:

- **Raspberry Pi camera** – With the apt packages installed above the Picamera2
  backend is available inside the virtual environment (thanks to
  `--system-site-packages`). No additional Python packages are required.
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

