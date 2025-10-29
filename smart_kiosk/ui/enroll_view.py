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
        self.video = None

        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(640, 480)
        self.thumb = QtWidgets.QLabel()
        self.thumb.setFixedSize(160, 160)
        self.thumb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet("background:#222;color:#ccc")

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

        # Right panel with thumbnail and hints
        self.hints = QtWidgets.QLabel(
            "Enroll Mode\n- Capture 5–10 times\n- Vary angle/lighting\n- Keep face centered\n- Ensure sharpness"
        )
        self.hints.setStyleSheet("color:#ccc")

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Last Capture:"))
        right.addWidget(self.thumb)
        right.addSpacing(8)
        right.addWidget(self.hints)
        right.addStretch(1)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.label, stretch=3)
        top.addLayout(right, stretch=1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(form)
        layout.addWidget(self.btn_capture)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.msg)

        self.templates: List[np.ndarray] = []
        self.required_captures = 5
        self.target_captures = 10
        self.btn_save.setEnabled(False)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        try:
            if self.video is not None:
                self.video.release()
        except Exception:
            pass
        return super().closeEvent(e)

    def showEvent(self, e: QtGui.QShowEvent) -> None:
        if self.video is None:
            self.video = self._open_camera(self.cfg)
        if not self.timer.isActive():
            self.timer.start(30)
        return super().showEvent(e)

    def hideEvent(self, e: QtGui.QHideEvent) -> None:
        if self.timer.isActive():
            self.timer.stop()
        if self.video is not None:
            try:
                self.video.release()
            except Exception:
                pass
            self.video = None
        return super().hideEvent(e)

    def _on_timer(self):
        ok, frame = self.video.read() if self.video is not None else (False, None)
        if ok:
            # Draw detection box for alignment help
            det = self.pipeline.detect_best(frame)
            disp = frame.copy()
            text = (
                f"Captured {len(self.templates)}/{self.target_captures}"
                f"  |  Need ≥{self.required_captures} before Save"
            )
            if det:
                x, y, w, h = det.bbox
                # Quality indicator
                q_ok = self.pipeline.quality_ok(frame, det)
                color = (0, 255, 0) if q_ok else (0, 0, 255)
                cv2.rectangle(disp, (x, y), (x + w, y + h), color, 2)
                if not q_ok:
                    text += "  (improve sharpness/center face)"
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
        target = pix.scaled(
            self.label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(target)

    def on_capture(self):
        ok, frame = self.video.read()
        if not ok:
            self.msg.setText("Camera error")
            return
        if len(self.templates) >= self.target_captures:
            self.msg.setText(f"Reached target of {self.target_captures} captures")
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
        # Show last captured thumbnail (aligned face)
        aligned_rgb = self.pipeline.aligner.align(frame, det)
        thumb = QtGui.QImage(
            aligned_rgb.data,
            aligned_rgb.shape[1],
            aligned_rgb.shape[0],
            aligned_rgb.shape[1] * 3,
            QtGui.QImage.Format.Format_RGB888,
        )
        self.thumb.setPixmap(QtGui.QPixmap.fromImage(thumb).scaled(self.thumb.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self.msg.setText(f"Captured {len(self.templates)}/{self.target_captures}")
        self._update_buttons()

    def on_save(self):
        emp_id = self.e_id.text().strip()
        if not emp_id:
            self.msg.setText("Employee ID required")
            return
        if len(self.templates) < self.required_captures:
            self.msg.setText(f"Please capture at least {self.required_captures} templates")
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
        self._update_buttons()

    def _update_buttons(self):
        self.btn_save.setEnabled(len(self.templates) >= self.required_captures)
        # Optional: disable capture when target reached
        self.btn_capture.setEnabled(len(self.templates) < self.target_captures)

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
