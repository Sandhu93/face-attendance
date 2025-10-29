from typing import Optional
import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from smart_kiosk.pipeline.face_pipeline import FacePipeline
from smart_kiosk.app.config import AppConfig
from smart_kiosk.db import Database


class AttendanceView(QtWidgets.QWidget):
    punch = QtCore.pyqtSignal(str, str, float, float)

    def __init__(self, cfg: AppConfig, db: Database, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.db = db
        self.pipeline = FacePipeline(cfg, db)

        self.video = cv2.VideoCapture(cfg.camera.device_index)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
        self.video.set(cv2.CAP_PROP_FPS, cfg.camera.fps)

        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(30)

        self.last_employee: Optional[str] = None

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        try:
            self.video.release()
        except Exception:
            pass
        return super().closeEvent(e)

    def _on_timer(self):
        ok, frame = self.video.read()
        if not ok:
            return
        res = self.pipeline.identify(frame)
        disp = frame.copy()
        if res.det is not None:
            x, y, w, h = res.det.bbox
            color = (0, 255, 0) if res.is_live else (0, 0, 255)
            cv2.rectangle(disp, (x, y), (x + w, y + h), color, 2)
        text = "Scanning..."
        if res.employee_id and res.is_live and res.score <= self.cfg.thresholds.match_threshold:
            last = self.db.last_event_type_today(res.employee_id)
            event_type = "OUT" if last == "IN" else "IN"
            self.db.add_event(res.employee_id, event_type, 1.0 - res.score, res.liveness, self.cfg.device.device_id)
            name = next((e['name'] for e in self.db.list_employees() if e['id'] == res.employee_id), res.employee_id)
            text = f"{name}: {event_type} (sim={1.0 - res.score:.2f}, live={res.liveness:.2f})"
        elif res.is_live and res.det is not None:
            text = f"Unknown face (live={res.liveness:.2f})"
        elif res.det is not None:
            text = f"Rejected: liveness {res.liveness:.2f}"
        self._set_pixmap(disp, text)

    def _set_pixmap(self, frame_bgr: np.ndarray, text: str):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        painter = QtGui.QPainter(pix)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        painter.setFont(QtGui.QFont("Arial", 18))
        painter.drawText(20, 40, text)
        painter.end()
        self.label.setPixmap(pix)

