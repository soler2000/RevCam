PR Installation Guide
======================

This document provides a copy-paste friendly checklist for installing and running
this pull request build of RevCam on a Raspberry Pi running Raspberry Pi OS
(Bookworm). Follow the steps in order to ensure Picamera2 and other native
dependencies are available to the application.

Prerequisites
-------------
* Raspberry Pi OS (Bookworm) with camera firmware enabled via `raspi-config`.
* Internet connectivity (APT + GitHub + PiWheels).
* Python 3.11 (the default on Raspberry Pi OS Bookworm).

1. Update the base system and install Raspberry Pi camera packages
-----------------------------------------------------------------
```bash
sudo apt update
sudo apt install -y \
  git python3 python3-venv python3-pip \
  python3-picamera2 python3-prctl python3-simplejpeg \
  libatlas-base-dev \
  libjpeg-dev zlib1g-dev pkg-config
```
These packages provide the official Picamera2 stack alongside the JPEG encoder
used by the streaming pipeline. The same set is available through
`./scripts/install_prereqs.sh` if you prefer a reusable helper.

2. Clone (or update) the RevCam repository
-----------------------------------------
```bash
REVCAM_REPO="https://github.com/soler2000/RevCam.git"
REVCAM_REF="pull/<PR_NUMBER>/head"  # replace <PR_NUMBER> with the PR you want to test

if [ ! -d RevCam ]; then
  git clone "$REVCAM_REPO" RevCam
fi

cd RevCam

git fetch "$REVCAM_REPO" "$REVCAM_REF"
git checkout FETCH_HEAD
```
Replace `REVCAM_REF` with the branch or PR ref you want to test (for example,
`pull/123/head`).

3. Create a virtual environment with system Picamera2 access
-----------------------------------------------------------
```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
The `--system-site-packages` flag exposes the APT-installed Picamera2 modules to
the virtual environment so RevCam can import them without rebuilding.

4. Install RevCam and Python dependencies
----------------------------------------
```bash
pip install --prefer-binary --extra-index-url https://www.piwheels.org/simple -e .
```
The PiWheels index provides pre-built ARM wheels for heavy packages. When a
wheel is not available, the system libraries installed in step 1 allow `pip` to
build the modules locally.

5. Verify camera availability before launching
----------------------------------------------
Busy cameras (often caused by the legacy MMAL stack) will prevent Picamera2 from
starting. Run the diagnostics helper to confirm the device is free:
```bash
python -m rev_cam.diagnostics
```
If the output lists `kworker/R-mmal-vchiq` threads or other processes, disable
the legacy camera interface using `sudo raspi-config` → Interface Options →
Legacy Camera (set to *Disabled*), remove `start_x=1` from `/boot/config.txt`,
and reboot. Stop any conflicting services (for example, `libcamera-vid` or
`libcamera-still`) before launching RevCam.

6. Launch the RevCam server
---------------------------
```bash
uvicorn rev_cam.app:create_app --factory --host 0.0.0.0 --port 9000
```
Open `http://<pi-ip>:9000` in your browser. The settings page exposes camera
source and resolution dropdowns—select *PiCamera2* to use the hardware camera
and choose a resolution that fits your bandwidth and display needs once
diagnostics report the device as available.

7. Optional: enable the service on boot
--------------------------------------
Follow the instructions in `README.md` to create a systemd service or use a
process manager of your choice. Ensure the service activates the virtual
environment before launching `uvicorn`.

Troubleshooting
---------------
* **`ModuleNotFoundError: picamera2`** – confirm steps 1 and 3 were executed so
  the virtual environment can see the system Picamera2 packages.
* **Busy camera errors** – re-run `python -m rev_cam.diagnostics` to identify
  lingering processes. Disable the legacy camera interface and reboot if
  `kworker/R-mmal-vchiq` threads are reported.
* **`simplejpeg` missing** – re-run step 4. If installation fails, verify the
  APT packages listed in step 1 are installed.
