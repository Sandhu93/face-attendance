import os
from smart_kiosk.db import Database


def test_db_crud(tmp_path):
    db_path = os.path.join(tmp_path, "test.db")
    db = Database(db_path)
    db.upsert_employee("e1", "Alice")
    emps = db.list_employees()
    assert any(e["id"] == "e1" for e in emps)
    # Add a dummy template
    tid = db.add_template("e1", b"abcd", 1, 0.9)
    assert isinstance(tid, str)
    # Add an event
    eid = db.add_event("e1", "IN", 0.9, 0.9, "dev1")
    assert isinstance(eid, str)

