# RevCam

RevCam is a low-latency reversing camera stack targeted at the Raspberry Pi Zero 2 W (Bookworm).
It streams a live camera feed to iOS devices using an MJPEG pipeline and exposes a settings panel
for configuring image orientation and capture resolution. The processing pipeline is modular so
that future driver-assistance overlays can be injected on the server without major changes.

## Features

- Fast MJPEG video delivery with a dedicated low-latency broadcaster tuned for
  mobile Safari (iPhone/iPad), plus pause/resume controls and automatic
  reconnect behaviour.
- Camera orientation controls (rotation and horizontal/vertical flips) and preset resolution
  options for balancing clarity against bandwidth.
- Modular frame processing pipeline ready for future overlays (e.g. guidelines).
- REST API for orientation control and camera management.
- Optional battery indicator when an INA219 sensor is connected, showing live
  percentage, voltage, and current draw in the viewer.
- One-tap snapshot capture via REST endpoint and the live viewer download
  control.
- Built-in mDNS advertisement so `motion.local:9000` resolves on iOS clients
  even when the RevCam hotspot is active.

## Project layout

```
src/
  rev_cam/
    __init__.py
    app.py                # FastAPI application and routes
    camera.py             # Camera source abstractions
    config.py             # Orientation persistence and validation
    pipeline.py           # Frame processing pipeline (orientation + overlays)
    streaming.py          # MJPEG streaming helpers
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
   sudo apt install python3-picamera2 python3-prctl python3-simplejpeg
  ```

   Prefer a single command? Run the bundled helper, which also installs
   the native JPEG encoder used by the streaming pipeline:

   > **Tip:** The SimpleJPEG package is published as `python3-simplejpeg` on
   > Raspberry Pi OS. Running `sudo apt install simplejpeg` will fail with an
   > "unable to locate package" error. If the package is unavailable from your
   > mirror, install SimpleJPEG from PyPI after activating the virtual
   > environment (see the next step).

   ```bash
   ./scripts/install_prereqs.sh
   ```

2. Create a virtual environment that can see the system packages you just
   installed (this prevents `pip` from trying to rebuild Picamera2 and PiDNG):

   ```bash
   python3 -m venv --system-site-packages .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

   If APT could not find `python3-simplejpeg`, install the wheel inside the
   active virtual environment before proceeding:

   ```bash
   pip install --prefer-binary --extra-index-url https://www.piwheels.org/simple simplejpeg
   ```

3. Install RevCam from the source tree. On Pi hardware this step pulls
   pre-built wheels from PiWheels, so it usually completes quickly. Seeing a
   long list ending with `Successfully installed ...` means the step finished
   successfully. You can either run the convenience script:

   ```bash
   ./scripts/install.sh --pi
   ```

   or execute `pip` manually:

   ```bash
   pip install --prefer-binary --extra-index-url https://www.piwheels.org/simple -e .
   ```

The `--prefer-binary` flag asks `pip` to fetch pre-built wheels when
available, and the PiWheels index provides ARM builds for most dependencies.

### LED ring requirements

RevCam's status indicator uses a 16-pixel WS2812/NeoPixel ring attached to GPIO18.
Driving the LEDs relies on Adafruit's CircuitPython stack. The installation
script automatically upgrades the necessary packages after the core project is
installed, but when managing the environment manually be sure to install them
yourself:

```bash
pip install --upgrade adafruit-blinka adafruit-circuitpython-neopixel
```

Without these modules the LED helper falls back to a no-op driver and the ring
remains dark.

Adafruit's WS2812 driver accesses the Pi's DMA engine via ``/dev/mem`` so it
requires root privileges. When RevCam starts without ``sudo`` (or an
equivalent capability such as the ``pigpiod`` backend) the helper logs "LED
driver unavailable" and disables the animations. Run Uvicorn with ``sudo`` or
configure an alternative backend before expecting the ring to light up.

### Saving snapshots

The live view footer includes a **Save snapshot** button that fetches a still
image without interrupting the MJPEG stream. RevCam exposes the same capture
functionality via two HTTP endpoints:

- `GET /api/camera/snapshot` returns a JPEG with headers suitable for API
  clients and short-term caching disabled.
- `GET /snapshot.jpg` provides the same image with a friendlier path for direct
  browser access.

Both endpoints run the captured frame through the configured pipeline and
return the processed output, so overlays and orientation adjustments are
preserved in the saved image.

### Battery monitoring requirements

Battery telemetry is optional. To enable it connect an INA219 current/voltage
sensor to the Pi's I²C bus and install the CircuitPython driver inside the
RevCam virtual environment:

```bash
pip install adafruit-blinka adafruit-circuitpython-ina219
```

Once available the live view automatically polls the sensor and displays the
pack percentage, voltage, and current draw in the header.

If your sensor is attached to a non-default Linux I²C bus, set the
`REVCAM_I2C_BUS` environment variable before starting RevCam. For example, to
use bus 29:

```bash
export REVCAM_I2C_BUS=29
uvicorn rev_cam.app:create_app --factory --host 0.0.0.0 --port 9000
```

When overriding the bus number install the optional
`adafruit-circuitpython-extended-bus` package so RevCam can open the desired
adapter.

RevCam expects the INA219 to respond at address `0x43`. If the sensor has been
configured differently adjust the jumper configuration accordingly.

### Distance monitoring requirements

The VL53L1X time-of-flight sensor powers the distance overlay. Install the
CircuitPython dependencies inside the same environment as RevCam:

```bash
pip install adafruit-blinka adafruit-circuitpython-vl53l1x
```

With those packages missing the distance panel reports **driver unavailable**
when it cannot import the VL53L1X driver or supporting `board` module.

Just like the INA219, the VL53L1X defaults to the Pi's primary I²C controller at
address `0x29`. When connecting the sensor to a different bus export
`REVCAM_I2C_BUS` before launching the server and install the optional
`adafruit-circuitpython-extended-bus` helper so RevCam can open the alternate
adapter.

### Development machine installation

For local development on non-Pi machines a regular virtual environment is
sufficient:

```bash
./scripts/install.sh --dev
```

Under the hood the script creates a `.venv` virtual environment and installs the
project in editable mode. If you prefer to perform the steps manually, run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### NetworkManager permissions for Wi-Fi control

RevCam drives Wi-Fi and hotspot features through NetworkManager's `nmcli`
command-line tool. The user account running RevCam therefore needs permission to
modify Wi-Fi connections and toggle hotspots. When running inside a service
account (for example under systemd) NetworkManager often denies these actions
with errors such as:

```
Error: Failed to setup a Wi-Fi hotspot: Not authorized to control networking.
```

To grant the necessary access either run RevCam as a user that already has
network privileges (e.g. the default `pi` account on Raspberry Pi OS) or add a
Polkit rule allowing your service account to control NetworkManager. A minimal
rule that authorises members of the `revcam` group looks like this:

```bash
sudo groupadd -f revcam
sudo usermod -aG revcam <your-service-user>
sudo tee /etc/polkit-1/rules.d/10-revcam-nm.rules >/dev/null <<'EOF'
polkit.addRule(function(action, subject) {
  if (action.id.indexOf('org.freedesktop.NetworkManager.') === 0 &&
      subject.isInGroup('revcam')) {
    return polkit.Result.YES;
  }
});
EOF
```

After reloading the service (or rebooting) `nmcli general permissions` should
list `yes` for the `org.freedesktop.NetworkManager.*` capabilities and RevCam's
hotspot toggle will succeed.

### Hotspot hostname and development-mode rollback

When the RevCam hotspot is enabled the server now advertises itself over mDNS
as `motion.local`, so iOS devices can simply browse to
[`http://motion.local:9000/`](http://motion.local:9000/) without looking up the
hotspot's IP address. If development mode is enabled for the hotspot, RevCam
waits up to 120 seconds for the access point to become fully active before
rolling back to the previous Wi-Fi profile, giving clients more time to join the
new network. The announcer prefers the pure-Python `zeroconf` package (installed
automatically when you run `pip install -e .`), but also falls back to Avahi's
`avahi-publish` CLI. If you upgraded an existing environment, make sure
`zeroconf` is installed or add `avahi-utils` via `sudo apt install avahi-utils`
so `motion.local` resolves while the hotspot is active.

### Copy-paste bootstrap script (development machines)

Need a single snippet you can drop into a terminal? The commands below clone (or
update) the repository and prepare a development virtual environment. Replace
the `REVCAM_REPO` value if you use a different remote. On Raspberry Pi hardware
use the dedicated script in the next section so the virtual environment can see
the system Picamera2 packages.

```bash
REVCAM_REPO="https://github.com/soler2000/RevCam.git"

set -euo pipefail

if [ ! -d RevCam ]; then
  git clone "$REVCAM_REPO" RevCam
else
  git -C RevCam pull --ff-only
fi

cd RevCam
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

### Raspberry Pi copy-paste bootstrap script

Need a single snippet tuned for Raspberry Pi OS? The commands below install the
apt-packaged Picamera2 stack, clone (or update) the repository, create a virtual
environment that can see the system packages, install RevCam using PiWheels, and
launch the server on port 9000.

```bash
#!/usr/bin/env bash
set -euo pipefail

REVCAM_REPO="https://github.com/soler2000/RevCam.git"
REVCAM_REF="main"  # replace with a branch or pull/<ID>/head for PR testing

sudo apt update
sudo apt install -y python3-picamera2 python3-prctl python3-simplejpeg

# If python3-simplejpeg is unavailable, install it from PyPI after the
# virtual environment is created:
#   source .venv/bin/activate
#   pip install --prefer-binary --extra-index-url https://www.piwheels.org/simple simplejpeg

if [ ! -d RevCam ]; then
  git clone "$REVCAM_REPO" RevCam
fi

cd RevCam
git fetch "$REVCAM_REPO" "$REVCAM_REF"
git checkout FETCH_HEAD

# Install runtime deps; use --dev on non-Pi development machines
./scripts/install.sh --pi

uvicorn rev_cam.app:create_app --factory --host 0.0.0.0 --port 9000
```

The script automatically reuses the existing virtual environment on subsequent
runs. Pass `--recreate` to `scripts/install.sh` if you want to start with a
fresh environment.

### Copy-paste branch runner

Need to try out a feature branch from GitHub? The snippet below checks out the
branch, ensures the virtual environment exists, installs dependencies, and
launches the development server. Replace `REVCAM_BRANCH` with the branch you
want to test.

```bash
REVCAM_REPO="https://github.com/soler2000/RevCam.git"
REVCAM_BRANCH="main"
```

```bash
set -euo pipefail

if [ ! -d RevCam ]; then
  git clone "$REVCAM_REPO" RevCam
fi

cd RevCam
git fetch origin "$REVCAM_BRANCH"
git checkout "$REVCAM_BRANCH"

./scripts/install.sh --pi

uvicorn rev_cam.app:create_app --factory --host 0.0.0.0 --port 9000
```

### Fast branch updates without rebuilding dependencies

If you frequently test branches that rebuild native packages (for example
`numpy`), use `git worktree` to reuse the same repository clone and
virtual environment. Build the dependencies once on your main checkout, then
create lightweight worktrees for feature branches:

```bash
cd RevCam
source .venv/bin/activate
git fetch origin my-feature
git worktree add ../RevCam-my-feature origin/my-feature
```

The new directory (`../RevCam-my-feature`) shares the original `.git` folder, so
you can activate the already-initialised virtual environment instead of
rebuilding wheels:

```bash
cd ../RevCam-my-feature
source ../RevCam/.venv/bin/activate
pip install --no-deps -e .
uvicorn rev_cam.app:create_app --factory --host 0.0.0.0 --port 9000
```

Use `pip install -e .[dev]` only when `pyproject.toml` changes; otherwise
`--no-deps` reuses the compiled dependencies from the shared virtual
environment. When you're finished with a branch, clean up the worktree with
`git worktree remove ../RevCam-my-feature`.

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
and applied immediately to the outgoing MJPEG stream.

## Testing

```
pytest
```

Unit tests cover the frame pipeline and configuration management so that critical
behaviour remains reliable.

### Troubleshooting

#### Picamera2 module cannot be imported

If you see `ModuleNotFoundError: picamera2`, the virtual environment cannot see
the Raspberry Pi OS Picamera2 packages. Ensure the packages are installed
(`sudo apt install python3-picamera2 python3-prctl`) and recreate the virtual
environment with system packages enabled:

```bash
rm -rf .venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
./scripts/install.sh --pi
```

Alternatively rerun `./scripts/install.sh --pi --recreate` to automate the
steps.

#### Camera reports "Device or resource busy"

Another process is still using the camera. Stop conflicting services such as
`libcamera-apps` and close any applications accessing the device. If the error
mentions kernel threads named `kworker/R-mmal-vchiq`, the legacy camera
interface is enabled. Disable it via `sudo raspi-config` (Interface Options →
Legacy Camera) or remove `start_x=1` from `/boot/config.txt`, then reboot.

Need a quick status report? Run the diagnostics helper to print detected
services and processes:

```bash
python -m rev_cam.diagnostics
```

#### Uvicorn reports "address already in use"

Another RevCam instance is already bound to the requested port. Stop the
existing process (`Ctrl+C` in the other terminal or `pkill -f uvicorn`) or use a
different port with `--port`.

## Future work

- Integrate computer-vision overlays (parking guidelines, obstacle detection).
- Replace the default software encoder with hardware acceleration.
- Add authentication for configuration endpoints.

