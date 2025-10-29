from PyQt6 import QtCore, QtWidgets
from smart_kiosk.app.config import load_config, AppConfig
from smart_kiosk.db import Database
from .attendance_view import AttendanceView
from .enroll_view import EnrollView


class KioskApp(QtWidgets.QMainWindow):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Smart Kiosk - Face Attendance")
        self.setCursor(QtCore.Qt.CursorShape.BlankCursor)
        self.showFullScreen()

        self.db = Database(cfg.paths.db_path)

        self.stack = QtWidgets.QStackedWidget()
        self.attendance = AttendanceView(cfg, self.db)
        self.enroll = EnrollView(cfg, self.db)
        self.stack.addWidget(self.attendance)
        self.stack.addWidget(self.enroll)
        self.setCentralWidget(self.stack)

        # Simple alternating via F1/F2 (admin PIN should gate in production)
        from PyQt6 import QtGui
        QtGui.QShortcut(QtGui.QKeySequence("F1"), self, activated=lambda: self.stack.setCurrentIndex(0))
        QtGui.QShortcut(QtGui.QKeySequence("F2"), self, activated=lambda: self.stack.setCurrentIndex(1))


def run():
    from PyQt6 import QtWidgets

    cfg = load_config()
    app = QtWidgets.QApplication([])
    win = KioskApp(cfg)
    win.show()
    app.exec()
