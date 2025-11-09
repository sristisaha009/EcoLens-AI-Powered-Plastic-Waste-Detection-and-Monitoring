# db.py
import sqlite3
from datetime import datetime
from typing import Optional, Tuple, List, Dict

DB_PATH = "reports.db"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    lat REAL,
    lon REAL,
    categories TEXT,        -- JSON string: {"plastic_bottle":2,...}
    severity REAL,          -- numeric severity/score
    notes TEXT
);
"""

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_conn()
    conn.execute(CREATE_SQL)
    conn.commit()
    conn.close()

def add_report(filename: str, lat: Optional[float], lon: Optional[float],
               categories: str, severity: float, notes: Optional[str] = "") -> int:
    conn = get_conn()
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO reports (filename,timestamp,lat,lon,categories,severity,notes) VALUES (?,?,?,?,?,?,?)",
                (filename, ts, lat, lon, categories, severity, notes))
    conn.commit()
    _id = cur.lastrowid
    conn.close()
    return _id

def get_all_reports() -> List[Dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id,filename,timestamp,lat,lon,categories,severity,notes FROM reports ORDER BY timestamp DESC")
    rows = cur.fetchall()
    cols = ["id","filename","timestamp","lat","lon","categories","severity","notes"]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def get_summary(filter_categories=None):
    """
    Returns total_reports, avg_severity, and category_counts.
    If filter_categories is provided (list), only include reports matching these categories.
    """
    import sqlite3
    import json

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT categories, severity FROM reports")
    rows = cursor.fetchall()

    total_reports = 0
    severity_sum = 0
    category_counts = {}

    for cat_json, sev in rows:
        try:
            cats = json.loads(cat_json) if cat_json else {}
        except json.JSONDecodeError:
            cats = {}

        if filter_categories:
            cats = {k: v for k, v in cats.items() if k in filter_categories}
            if not cats:
                continue  # skip reports with no matching category

        total_reports += 1
        severity_sum += sev
        for k, v in cats.items():
            category_counts[k] = category_counts.get(k, 0) + v

    avg_severity = severity_sum / total_reports if total_reports > 0 else 0
    conn.close()

    return {"total_reports": total_reports, "avg_severity": avg_severity, "category_counts": category_counts}
