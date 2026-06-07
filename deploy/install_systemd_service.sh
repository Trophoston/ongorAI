#!/usr/bin/env bash
set -euo pipefail

# Install Ong-Or FastAPI systemd unit on Linux board.
# Usage:
#   sudo bash deploy/install_systemd_service.sh \
#       --user ongor \
#       --group ongor \
#       --workdir /opt/ongor \
#       --python /usr/bin/python3

SERVICE_NAME="ongor-fastapi"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SRC="${SCRIPT_DIR}/ongor-fastapi.service"
UNIT_DST="/etc/systemd/system/${SERVICE_NAME}.service"
ENV_DST="/etc/default/${SERVICE_NAME}"
ENV_EXAMPLE="${SCRIPT_DIR}/ongor-fastapi.env.example"

APP_USER="ongor"
APP_GROUP="ongor"
WORKDIR="/opt/ongorAI"
PYTHON_BIN="/usr/bin/python3"
VENV_DIR="${WORKDIR}/.venv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      APP_USER="$2"; shift 2 ;;
    --group)
      APP_GROUP="$2"; shift 2 ;;
    --workdir)
      WORKDIR="$2"; shift 2 ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1 ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (use sudo)." >&2
  exit 1
fi

if [[ ! -f "$UNIT_SRC" ]]; then
  echo "Missing ${UNIT_SRC}" >&2
  exit 1
fi

if [[ ! -d "$WORKDIR" ]]; then
  echo "Workdir not found: $WORKDIR" >&2
  echo "Clone/copy project to this path first." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -m venv "$VENV_DIR" >/dev/null 2>&1; then
  echo "Failed to create virtualenv at ${VENV_DIR}." >&2
  echo "Install the board package that provides python3-venv, then run this script again." >&2
  exit 1
fi

if ! "$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null 2>&1; then
  echo "Failed to bootstrap pip inside ${VENV_DIR}." >&2
  echo "Install python3-venv / ensurepip support, then run this script again." >&2
  exit 1
fi

if ! "$VENV_DIR/bin/python" -m pip install -r "${SCRIPT_DIR}/../requirements.txt"; then
  echo "Failed to install Python requirements into ${VENV_DIR}." >&2
  echo "Check network access and package compatibility, then try again." >&2
  exit 1
fi

# Ensure service user/group exist.
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  useradd -r -s /usr/sbin/nologin "$APP_USER"
fi
if ! getent group "$APP_GROUP" >/dev/null 2>&1; then
  groupadd -r "$APP_GROUP"
fi
usermod -a -G "$APP_GROUP" "$APP_USER" || true

# Install service file with board-specific substitutions.
sed \
  -e "s|^User=.*|User=${APP_USER}|" \
  -e "s|^Group=.*|Group=${APP_GROUP}|" \
  -e "s|^WorkingDirectory=.*|WorkingDirectory=${WORKDIR}|" \
  -e "s|^ExecStart=.*|ExecStart=${VENV_DIR}/bin/python -m uvicorn api.main:app --host ${ONGOR_HOST:-0.0.0.0} --port ${ONGOR_PORT:-8000}|" \
  "$UNIT_SRC" > "$UNIT_DST"

# Install env file if missing.
if [[ ! -f "$ENV_DST" ]]; then
  cp "$ENV_EXAMPLE" "$ENV_DST"
fi

chown -R "$APP_USER":"$APP_GROUP" "$WORKDIR"

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo "Installed and started ${SERVICE_NAME}."
echo "Edit env: ${ENV_DST}"
echo "Check status: systemctl status ${SERVICE_NAME} --no-pager"
echo "Logs: journalctl -u ${SERVICE_NAME} -f"
