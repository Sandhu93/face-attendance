from typing import List
import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from smart_kiosk.pipeline.face_pipeline import FacePipeline
from smart_kiosk.app.config import AppConfig
from smart_kiosk.db import Database


class EnrollView(QtWidgets.QWidget):
    def __init__(self, cfg: AppConfig, db: Database, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.db = db
        self.pipeline = FacePipeline(cfg, db)
        self.video = self._open_camera(cfg)

        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        form = QtWidgets.QFormLayout()
        self.e_id = QtWidgets.QLineEdit()
        self.e_name = QtWidgets.QLineEdit()
        self.e_dept = QtWidgets.QLineEdit()
        self.e_email = QtWidgets.QLineEdit()
        self.e_phone = QtWidgets.QLineEdit()
        form.addRow("Employee ID", self.e_id)
        form.addRow("Name", self.e_name)
        form.addRow("Dept", self.e_dept)
        form.addRow("Email", self.e_email)
        form.addRow("Phone", self.e_phone)

        self.btn_capture = QtWidgets.QPushButton("Capture Template")
        self.btn_save = QtWidgets.QPushButton("Save Employee")
        self.btn_capture.clicked.connect(self.on_capture)
        self.btn_save.clicked.connect(self.on_save)

        self.msg = QtWidgets.QLabel()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addLayout(form)
        layout.addWidget(self.btn_capture)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.msg)

        self.templates: List[np.ndarray] = []

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(30)

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        try:
            self.video.release()
        except Exception:
            pass
        return super().closeEvent(e)

    def _on_timer(self):
        ok, frame = self.video.read()
        if ok:
            self._set_pixmap(frame, f"Templates: {len(self.templates)}")

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

    def on_capture(self):
        ok, frame = self.video.read()
        if not ok:
            self.msg.setText("Camera error")
            return
        det = self.pipeline.detect_best(frame)
        if not det or not self.pipeline.quality_ok(frame, det):
            self.msg.setText("Face quality too low")
            return
        liv = self.pipeline.liveness.passive(frame, det)
        if not liv.is_live:
            self.msg.setText(f"Liveness failed: {liv.explanation}")
            return
        emb = self.pipeline.embed(frame, det)
        self.templates.append(emb)
        self.msg.setText("Captured")

    def on_save(self):
        emp_id = self.e_id.text().strip()
        if not emp_id:
            self.msg.setText("Employee ID required")
            return
        name = self.e_name.text().strip() or emp_id
        self.db.upsert_employee(emp_id, name, self.e_dept.text().strip(), self.e_email.text().strip(), self.e_phone.text().strip())
        # Store all captured templates
        for emb in self.templates:
            from smart_kiosk.app.utils import compress_embedding

            blob = compress_embedding(emb)
            self.db.add_template(emp_id, blob, version=1, quality=1.0)
        self.templates.clear()
        self.msg.setText("Saved employee")

    def _open_camera(self, cfg: AppConfig):
        cap = cv2.VideoCapture(cfg.camera.device_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
        cap = cv2.VideoCapture(cfg.camera.device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)
        return cap
