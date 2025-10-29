#!/usr/bin/env bash
set -euo pipefail

# Smart Kiosk install script for Raspberry Pi OS 64-bit
# - Installs Python deps
# - Creates data dirs
# - Installs systemd services (optional)

APP_DIR="/opt/smart-kiosk"
CFG_DIR="/etc/smart-kiosk"
DATA_DIR="/var/lib/smart-kiosk"

echo "[+] Creating directories"
sudo mkdir -p "$APP_DIR" "$CFG_DIR" "$DATA_DIR"
sudo chown -R $USER:"$USER" "$APP_DIR" "$CFG_DIR" "$DATA_DIR"

echo "[+] Copying application (adjust path as needed)"
# rsync -a ./ "$APP_DIR"/

echo "[+] Installing Python dependencies"
python3 -m pip install -r requirements.txt

if ! [ -f "$CFG_DIR/config.yaml" ]; then
  cat > "$CFG_DIR/config.yaml" <<EOF
camera:
  device_index: 0
  width: 1280
  height: 720
backend:
  face_backend: opencv
thresholds:
  match_threshold: 0.48
  liveness_threshold: 0.6
paths:
  data_dir: $DATA_DIR
  db_path: $DATA_DIR/attendance.db
device:
  device_id: $(hostname)
EOF
fi

echo "[+] Installing systemd services"
sudo cp system/kiosk.service /etc/systemd/system/kiosk.service
sudo cp system/kiosk-api.service /etc/systemd/system/kiosk-api.service
sudo systemctl daemon-reload
# sudo systemctl enable kiosk.service kiosk-api.service

echo "[+] Done. Configure at $CFG_DIR/config.yaml"

