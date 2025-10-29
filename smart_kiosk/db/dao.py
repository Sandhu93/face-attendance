import os
import sqlite3
import uuid
from typing import Any, Dict, List, Optional, Tuple
from smart_kiosk.app.utils import now_ts


def _read_schema() -> str:
    here = os.path.dirname(__file__)
    path = os.path.join(here, "schema.sql")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self):
        self.conn.executescript(_read_schema())
        self.conn.commit()

    def execute(self, sql: str, params: Tuple = ()):
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def query(self, sql: str, params: Tuple = ()) -> List[Tuple]:
        cur = self.conn.execute(sql, params)
        return cur.fetchall()

    # Employee ops
    def upsert_employee(self, id: str, name: str, dept: str = "", email: str = "", phone: str = "", active: int = 1):
        ts = now_ts()
        self.execute(
            """
            INSERT INTO employees (id,name,dept,email,phone,active,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                dept=excluded.dept,
                email=excluded.email,
                phone=excluded.phone,
                active=excluded.active,
                updated_at=excluded.updated_at
            """,
            (id, name, dept, email, phone, active, ts, ts),
        )

    def list_employees(self, active_only: bool = True) -> List[Dict[str, Any]]:
        rows = self.query(
            "SELECT id,name,dept,email,phone,active,created_at,updated_at FROM employees WHERE (?=0 OR active=1) ORDER BY name",
            (1 if active_only else 0,),
        )
        return [
            dict(
                id=r[0], name=r[1], dept=r[2], email=r[3], phone=r[4], active=r[5], created_at=r[6], updated_at=r[7]
            )
            for r in rows
        ]

    def add_template(self, employee_id: str, embedding_blob: bytes, version: int, quality: float) -> str:
        tid = str(uuid.uuid4())
        ts = now_ts()
        self.execute(
            "INSERT INTO face_templates (id,employee_id,embedding,version,quality,created_at) VALUES (?,?,?,?,?,?)",
            (tid, employee_id, embedding_blob, version, quality, ts),
        )
        return tid

    def get_templates(self, employee_id: Optional[str] = None) -> List[Tuple[str, str, bytes, int, float]]:
        if employee_id:
            rows = self.query(
                "SELECT id,employee_id,embedding,version,quality FROM face_templates WHERE employee_id=?",
                (employee_id,),
            )
        else:
            rows = self.query("SELECT id,employee_id,embedding,version,quality FROM face_templates")
        return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]

    def delete_employee(self, employee_id: str):
        self.execute("DELETE FROM employees WHERE id=?", (employee_id,))

    def add_event(self, employee_id: str, event_type: str, score: float, liveness: float, device_id: str, photo_path: str = "") -> str:
        eid = str(uuid.uuid4())
        ts = now_ts()
        self.execute(
            """
            INSERT INTO attendance_events (id,employee_id,event_type,score,liveness,device_id,created_at,photo_path)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (eid, employee_id, event_type, score, liveness, device_id, ts, photo_path),
        )
        return eid

    def last_event_type_today(self, employee_id: str) -> Optional[str]:
        # Midnight epoch for local day is not trivial without tz; use last 24h as approximation
        since = now_ts() - 24 * 3600
        rows = self.query(
            "SELECT event_type FROM attendance_events WHERE employee_id=? AND created_at>? ORDER BY created_at DESC LIMIT 1",
            (employee_id, since),
        )
        return rows[0][0] if rows else None
