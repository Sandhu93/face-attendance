import sys
import os
import csv
from typing import Optional
import typer
import uvicorn
from smart_kiosk.app.config import load_config
from smart_kiosk.db import Database


app = typer.Typer(name="kiosk")


@app.command()
def run():
    """Run the PyQt6 kiosk UI in fullscreen."""
    from smart_kiosk.ui.kiosk import run as run_ui

    run_ui()


@app.command()
def api(host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI service."""
    uvicorn.run("smart_kiosk.api.server:app", host=host, port=port, reload=False)


@app.command()
def export(csv_path: str = "events.csv"):
    """Export recent events to CSV."""
    cfg = load_config()
    db = Database(cfg.paths.db_path)
    rows = db.query(
        "SELECT id,employee_id,event_type,score,liveness,device_id,created_at,photo_path FROM attendance_events ORDER BY created_at DESC"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "employee_id", "event_type", "score", "liveness", "device_id", "created_at", "photo_path"])
        for r in rows:
            w.writerow(r)
    typer.echo(f"Exported {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    app()

