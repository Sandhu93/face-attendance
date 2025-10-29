import numpy as np
import os
from smart_kiosk.app.config import AppConfig
from smart_kiosk.db import Database
from smart_kiosk.pipeline.face_pipeline import FacePipeline


def test_identify_no_face(tmp_path):
    cfg = AppConfig()
    cfg.paths.db_path = os.path.join(tmp_path, "test.db")
    db = Database(cfg.paths.db_path)
    pipe = FacePipeline(cfg, db)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    res = pipe.identify(frame)
    assert res.employee_id is None

