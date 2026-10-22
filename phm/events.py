"""Local review ledger; records advice only, never equipment commands."""
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import pandas as pd
from phm.data import ROOT


def connect(path=None):
    path = Path(path or ROOT/'runs/events.sqlite'); path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, run_id TEXT, mission_id TEXT, timestamp_s REAL, predicted_label INTEGER, status TEXT, note TEXT, updated_at TEXT, UNIQUE(run_id, mission_id, timestamp_s))')
    return con


def save_event(run_id, mission, timestamp_s, predicted_label, status, note, path=None):
    if status not in {'待复核', '已确认', '误报', '已处理'}: raise ValueError('Unknown review state')
    with connect(path) as con:
        con.execute('INSERT INTO events(run_id,mission_id,timestamp_s,predicted_label,status,note,updated_at) VALUES (?,?,?,?,?,?,?) ON CONFLICT(run_id,mission_id,timestamp_s) DO UPDATE SET status=excluded.status,note=excluded.note,updated_at=excluded.updated_at',
                    (run_id, mission, float(timestamp_s), int(predicted_label), status, note, datetime.now(timezone.utc).isoformat()))


def read_events(path=None):
    with connect(path) as con: return pd.read_sql_query('SELECT * FROM events ORDER BY updated_at DESC', con)
