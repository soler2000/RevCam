#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${REVCAM_SERVICE_NAME:-revcam}"
if [[ "${SERVICE_NAME}" != *.service ]]; then
    SERVICE_NAME="${SERVICE_NAME}.service"
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${PROJECT_DIR}/scripts/run_with_sudo.sh"
if [[ ! -x "${RUNNER}" ]]; then
    echo "Expected helper ${RUNNER} to exist" >&2
    exit 1
fi

if [[ -n "${REVCAM_VENV:-}" ]]; then
    export VIRTUAL_ENV="${REVCAM_VENV}"
    export PATH="${REVCAM_VENV}/bin:${PATH}"
fi

SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

echo "Installing ${SERVICE_NAME} for RevCam from ${PROJECT_DIR}" >&2
sudo tee "${SERVICE_PATH}" > /dev/null <<UNIT
[Unit]
Description=RevCam reversing camera server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_DIR}
ExecStart=${RUNNER} --host 0.0.0.0 --port 9000
ExecStop=/bin/kill -s SIGINT \$MAINPID
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"

cat <<INFO
RevCam service installed.
Use './scripts/revcamctl.sh status' to verify the service state.
INFO
