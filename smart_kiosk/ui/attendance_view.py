from typing import Optional
import time
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

        self.video = None

        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(640, 480)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)

        self.last_employee: Optional[str] = None
        # Toast state for punch feedback
        self._toast_until: float = 0.0
        self._toast_text: str = ""
        self._toast_color: tuple[int, int, int] = (255, 255, 255)  # BGR
        self._toast_face: Optional[np.ndarray] = None
        self._last_punch_emp: Optional[str] = None
        self._last_punch_ts: float = 0.0

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        try:
            if self.video is not None:
                self.video.release()
        except Exception:
            pass
        return super().closeEvent(e)

    def showEvent(self, e: QtGui.QShowEvent) -> None:
        # Acquire camera when this view is shown
        if self.video is None:
            self.video = self._open_camera(self.cfg)
        if not self.timer.isActive():
            self.timer.start(30)
        return super().showEvent(e)

    def hideEvent(self, e: QtGui.QHideEvent) -> None:
        # Release camera to let other views use it
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
        if not ok:
            return
        now = time.time()
        disp = frame.copy()
        # If a toast is active, render it and skip detection to prevent double punches
        if self._toast_until > now:
            self._render_toast(disp)
            self._set_pixmap(disp, self._toast_text, pen_color=self._toast_color)
            return

        dets = self.pipeline.detect_all(frame)
        text = "Scanning..."
        recognized_results = []
        # Identify each detection and annotate
        for det in dets:
            res = self.pipeline.identify_det(frame, det)
            x, y, w, h = det.bbox
            color = (0, 255, 0) if (res.is_live and res.employee_id) else (0, 0, 255)
            cv2.rectangle(disp, (x, y), (x + w, y + h), color, 2)
            label = ""
            if res.is_live and res.employee_id and res.score <= self.cfg.thresholds.match_threshold:
                name = next((e['name'] for e in self.db.list_employees() if e['id'] == res.employee_id), res.employee_id)
                label = f"{name} ({1.0 - res.score:.2f})"
                recognized_results.append(res)
            elif res.is_live:
                label = f"Unknown ({res.liveness:.2f})"
            else:
                label = f"Not live ({res.liveness:.2f})"
            try:
                cv2.putText(disp, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                pass

        # Choose a single best recognition this frame for event/ toast handling
        best = None
        if recognized_results:
            best = min(recognized_results, key=lambda r: r.score)
        if best is not None:
            if self._last_punch_emp == best.employee_id and (now - self._last_punch_ts) < 3.0:
                self._set_pixmap(disp, text)
                return
            last = self.db.last_event_type_today(best.employee_id) if best.employee_id else None
            event_type = "OUT" if last == "IN" else "IN"
            # Persist event
            if best.employee_id:
                self.db.add_event(best.employee_id, event_type, 1.0 - best.score, best.liveness, self.cfg.device.device_id)
                self._last_punch_emp = best.employee_id
                self._last_punch_ts = now
                # Prepare toast data
                name = next((e['name'] for e in self.db.list_employees() if e['id'] == best.employee_id), best.employee_id)
                ts_str = QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
                self._toast_text = f"{name}  {event_type}  {ts_str}"
                self._toast_color = (0, 200, 0) if event_type == "IN" else (0, 0, 200)
                # Capture face thumbnail
                x, y, w, h = best.det.bbox if best.det is not None else (0, 0, 0, 0)
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(frame.shape[1], x + w)
                y1 = min(frame.shape[0], y + h)
                face = frame[y0:y1, x0:x1]
                self._toast_face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA) if face.size > 0 else None
                self._toast_until = now + 3.0
                self._render_toast(disp)
                self._set_pixmap(disp, self._toast_text, pen_color=self._toast_color)
                return
        self._set_pixmap(disp, text)

    def _set_pixmap(self, frame_bgr: np.ndarray, text: str, pen_color: tuple[int, int, int] | None = None):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        painter = QtGui.QPainter(pix)
        if pen_color is None:
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
        else:
            # pen_color is BGR; convert to RGB
            b, g, r = pen_color
            pen = QtGui.QPen(QtGui.QColor(int(r), int(g), int(b)))
        painter.setPen(pen)
        painter.setFont(QtGui.QFont("Arial", 18))
        painter.drawText(20, 40, text)
        painter.end()
        # Scale pixmap to fit label while keeping aspect ratio
        target = pix.scaled(
            self.label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(target)

    def _render_toast(self, frame_bgr: np.ndarray):
        # Draw colored border
        color = self._toast_color
        h, w = frame_bgr.shape[:2]
        cv2.rectangle(frame_bgr, (0, 0), (w - 1, h - 1), color, thickness=8)
        # Draw face thumbnail if available
        if self._toast_face is not None and self._toast_face.size > 0:
            face = self._toast_face
            fh, fw = face.shape[:2]
            x0, y0 = 20, 60
            x1, y1 = x0 + fw, y0 + fh
            if x1 < w and y1 < h:
                frame_bgr[y0:y1, x0:x1] = face

    def _open_camera(self, cfg: AppConfig):
        # Prefer V4L2 backend to avoid GStreamer issues on Pi
        cap = cv2.VideoCapture(cfg.camera.device_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)
        ok, _ = cap.read()
        if ok:
            return cap
        # Fallback to default backend
        cap.release()
        cap = cv2.VideoCapture(cfg.camera.device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)
        return cap
