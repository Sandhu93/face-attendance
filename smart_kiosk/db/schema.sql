PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS employees (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  dept TEXT,
  email TEXT,
  phone TEXT,
  active INTEGER NOT NULL DEFAULT 1,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS face_templates (
  id TEXT PRIMARY KEY,
  employee_id TEXT NOT NULL,
  embedding BLOB NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  quality REAL NOT NULL DEFAULT 0.0,
  created_at REAL NOT NULL,
  FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS attendance_events (
  id TEXT PRIMARY KEY,
  employee_id TEXT NOT NULL,
  event_type TEXT NOT NULL CHECK (event_type IN ('IN','OUT')),
  score REAL NOT NULL,
  liveness REAL NOT NULL,
  device_id TEXT NOT NULL,
  created_at REAL NOT NULL,
  photo_path TEXT,
  FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS devices (
  device_id TEXT PRIMARY KEY,
  location TEXT,
  tz TEXT,
  last_seen REAL
);

CREATE INDEX IF NOT EXISTS idx_templates_employee ON face_templates(employee_id);
CREATE INDEX IF NOT EXISTS idx_events_employee ON attendance_events(employee_id);
CREATE INDEX IF NOT EXISTS idx_events_created ON attendance_events(created_at);

