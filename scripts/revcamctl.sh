#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${REVCAM_SERVICE_NAME:-revcam}"
if [[ "${SERVICE_NAME}" != *.service ]]; then
    SERVICE_NAME="${SERVICE_NAME}.service"
fi

if [[ $# -lt 1 ]]; then
    cat <<USAGE
Usage: ${0##*/} <command>
Commands:
  status    Show current service status
  start     Start the RevCam service
  stop      Stop the RevCam service
  restart   Restart the RevCam service
  enable    Enable and start the service on boot
  disable   Disable and stop the service
USAGE
    exit 1
fi

command="$1"
shift || true

case "${command}" in
    status)
        sudo systemctl status "${SERVICE_NAME}" "$@"
        ;;
    start)
        sudo systemctl start "${SERVICE_NAME}" "$@"
        ;;
    stop)
        sudo systemctl stop "${SERVICE_NAME}" "$@"
        ;;
    restart)
        sudo systemctl restart "${SERVICE_NAME}" "$@"
        ;;
    enable)
        sudo systemctl enable --now "${SERVICE_NAME}" "$@"
        ;;
    disable)
        sudo systemctl disable --now "${SERVICE_NAME}" "$@"
        ;;
    *)
        echo "Unknown command: ${command}" >&2
        exit 2
        ;;
esac
