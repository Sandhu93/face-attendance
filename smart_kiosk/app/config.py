import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List


DEFAULT_CONFIG_PATHS = [
    os.environ.get("SMART_KIOSK_CONFIG", ""),
    ".smart-kiosk.yaml",
    "./config.yaml",
    "/etc/smart-kiosk/config.yaml",
]


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class BackendConfig:
    face_backend: str = "opencv"  # opencv|ncnn|onnx|edgetpu
    liveness_backend: str = "light"
    # Embedder backend: 'dct' (fallback) | 'onnx_arcface'
    embedder_backend: str = "dct"
    threads: int = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)


@dataclass
class Thresholds:
    # Tightened threshold for more robust identity separation
    match_threshold: float = 0.35  # cosine distance (1 - cosine_similarity)
    liveness_threshold: float = 0.6
    quality_sharpness_min: float = 25.0  # Reduced for angled faces
    min_face_px: int = 80  # Reduced minimum face size


@dataclass
class Paths:
    data_dir: str = os.path.abspath("./data")
    db_path: str = os.path.abspath("./data/attendance.db")
    photo_cache_dir: str = os.path.abspath("./data/photos")


@dataclass
class DeviceInfo:
    device_id: str = os.uname().nodename if hasattr(os, "uname") else os.getenv("COMPUTERNAME", "pi-device")
    location: str = "Office"
    tz: str = os.getenv("TZ", "UTC")


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    thresholds: Thresholds = field(default_factory=Thresholds)
    paths: Paths = field(default_factory=Paths)
    device: DeviceInfo = field(default_factory=DeviceInfo)
    audit_mode: bool = False
    allow_sync: bool = False
    sync_server_url: Optional[str] = None
    admin_pin: str = "1234"


def _merge_dict(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = _merge_dict(d[k], v)
        else:
            d[k] = v
    return d


def load_config() -> AppConfig:
    cfg = AppConfig()
    for p in [p for p in DEFAULT_CONFIG_PATHS if p]:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                base = cfg.__dict__.copy()
                merged = _merge_dict(base, data)
                # Manual map to dataclasses
                cfg.camera = CameraConfig(**merged.get("camera", {}))
                cfg.backend = BackendConfig(**merged.get("backend", {}))
                cfg.thresholds = Thresholds(**merged.get("thresholds", {}))
                cfg.paths = Paths(**merged.get("paths", {}))
                cfg.device = DeviceInfo(**merged.get("device", {}))
                cfg.audit_mode = bool(merged.get("audit_mode", cfg.audit_mode))
                cfg.allow_sync = bool(merged.get("allow_sync", cfg.allow_sync))
                cfg.sync_server_url = merged.get("sync_server_url", cfg.sync_server_url)
                cfg.admin_pin = merged.get("admin_pin", cfg.admin_pin)
            except Exception:
                # Fall back to defaults on parse errors
                pass
    # Ensure dirs
    os.makedirs(cfg.paths.data_dir, exist_ok=True)
    os.makedirs(cfg.paths.photo_cache_dir, exist_ok=True)
    return cfg
