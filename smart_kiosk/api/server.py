from typing import List, Optional
import base64
import io
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import numpy as np
import cv2

from smart_kiosk.app.config import load_config
from smart_kiosk.db import Database
from smart_kiosk.pipeline.face_pipeline import FacePipeline


app = FastAPI(title="Smart Kiosk API", version="0.1.0")
cfg = load_config()
db = Database(cfg.paths.db_path)
pipeline = FacePipeline(cfg, db)


class EnrollRequest(BaseModel):
    employee_id: str
    name: str
    dept: Optional[str] = ""
    images_b64: List[str]


@app.post("/enroll")
def enroll(req: EnrollRequest):
    db.upsert_employee(req.employee_id, req.name, req.dept or "")
    frames = []
    for b64 in req.images_b64:
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        frames.append(img)
    tids = pipeline.enroll_templates(req.employee_id, frames)
    return {"templates": tids}


class PunchResponse(BaseModel):
    employee_id: Optional[str]
    event_type: Optional[str]
    score: float
    liveness: float


@app.post("/punch", response_model=PunchResponse)
def punch(file: UploadFile = File(...)):
    raw = file.file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    res = pipeline.identify(img)
    if res.employee_id and res.is_live and res.score <= cfg.thresholds.match_threshold:
        last = db.last_event_type_today(res.employee_id)
        event_type = "OUT" if last == "IN" else "IN"
        db.add_event(res.employee_id, event_type, 1.0 - res.score, res.liveness, cfg.device.device_id)
        return PunchResponse(employee_id=res.employee_id, event_type=event_type, score=1.0 - res.score, liveness=res.liveness)
    return PunchResponse(employee_id=None, event_type=None, score=1.0 - res.score, liveness=res.liveness)


@app.get("/employees")
def employees():
    return db.list_employees(active_only=False)


@app.get("/events")
def events():
    rows = db.query("SELECT id,employee_id,event_type,score,liveness,device_id,created_at,photo_path FROM attendance_events ORDER BY created_at DESC LIMIT 500")
    return [
        dict(id=r[0], employee_id=r[1], event_type=r[2], score=r[3], liveness=r[4], device_id=r[5], created_at=r[6], photo_path=r[7])
        for r in rows
    ]

