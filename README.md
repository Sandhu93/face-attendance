Smart Kiosk: Face Attendance (Raspberry Pi)

Overview
- Offline, on-device face attendance system for Raspberry Pi 4/5 (64-bit).
- Private by design: DB stored locally; optional LAN sync.
- Modular backends: NCNN/ONNXRuntime/EdgeTPU (pluggable). Default fallback uses OpenCV-only pipeline to run without external models.

Key Features
- Attendance screen: detect → identify → liveness → mark IN/OUT automatically.
- Enrollment screen: add employee, capture templates with quality gating.
- Local FastAPI service: enroll, punch, list employees/events, sync.
- SQLite/SQLCipher DB with indices and compressed embeddings.
- Systemd services; config at /etc/smart-kiosk/config.yaml.

Status
- This repo is production-oriented and ready to run with an OpenCV fallback pipeline. For best accuracy and latency targets, drop in NCNN or EdgeTPU models and enable them in config.

Quick Start (Dev)
1) Python 3.11 recommended.
2) pip install -r requirements.txt
3) python -m smart_kiosk.cli run

Raspberry Pi Install
- Install system packages (PyQt6/OpenCV via apt):
  - `sudo apt update && sudo apt install -y python3-pyqt6 python3-opencv libgl1 libglib2.0-0 v4l-utils`
- If your desktop session is Wayland, install Wayland plugin or force XCB:
  - `sudo apt install -y qt6-wayland qt6-gtk-platformtheme`
  - Or set `QT_QPA_PLATFORM=xcb` when launching the app.
- Create venv that can see system packages:
  - `python3 -m venv --system-site-packages venv`
  - `source venv/bin/activate`
- Install Python deps (Pi-specific):
  - `pip install --upgrade pip`
  - `pip install -r requirements-pi.txt`
  - Note: requirements-pi pins `numpy<1.27` to avoid ABI conflicts with system `scipy` when using `--system-site-packages`.
- Configure and run:
  - `sudo mkdir -p /etc/smart-kiosk /var/lib/smart-kiosk && sudo chown -R $USER:$USER /var/lib/smart-kiosk`
  - Create `/etc/smart-kiosk/config.yaml` or set `SMART_KIOSK_CONFIG`
  - `SMART_KIOSK_CONFIG=/etc/smart-kiosk/config.yaml python -m smart_kiosk.cli run`
  - Tip: For OpenCV to prefer V4L2 (and avoid GStreamer warnings): `export OPENCV_VIDEOIO_PRIORITY_V4L2=1`


Project Layout
- app/: core config, utils, threading
- db/: SQLite access and migrations
- models/: inference backends (detector/embedding/liveness)
- pipeline/: face pipeline orchestration
- ui/: PyQt6 kiosk (attendance + enrollment)
- api/: FastAPI service
- scripts/: install, model conversion placeholders
- system/: systemd unit templates
- tests/: pytest unit/integration tests

Hardware/Performance
- Raspberry Pi 5 recommended; runs at 720p with OpenCV fallback at modest FPS; replace with NCNN or EdgeTPU for ≥25–35 FPS.
- USB3 UVC webcam.

Security & Privacy
- All processing is local. No cloud calls. Optional SQLCipher. Admin-gated enrollment and masked thumbnails by default.

Next Steps
- See config defaults in smart_kiosk/app/config.py; override with /etc/smart-kiosk/config.yaml.
- To enable OpenCV DNN detector, download the ResNet SSD model into `smart_kiosk/models/assets/` and set `backend.face_backend: opencv_dnn` in config.
  - Files required:
    - `deploy.prototxt`
    - `res10_300x300_ssd_iter_140000.caffemodel`
  - Example (on Pi):
    - `mkdir -p smart_kiosk/models/assets`
    - `wget -O smart_kiosk/models/assets/deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt`
    - `wget -O smart_kiosk/models/assets/res10_300x300_ssd_iter_140000.caffemodel https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel`
  - Then set in `/etc/smart-kiosk/config.yaml`:
    ```
    backend:
      face_backend: opencv_dnn
    ```
  - The code falls back to Haar if the files are missing.

License
Apache-2.0
