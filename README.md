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
- To enable NCNN/EdgeTPU, place models under models/assets/ and set backend in config.

License
Apache-2.0
