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


def incident_connection(path=None):
    con=connect(path)
    con.execute('CREATE TABLE IF NOT EXISTS incidents (incident_key TEXT PRIMARY KEY, run_id TEXT, event_id TEXT, mission_id TEXT, start_s REAL, confirmed_s REAL, end_s REAL, status TEXT, note TEXT, updated_at TEXT)')
    con.execute('CREATE TABLE IF NOT EXISTS incident_history (id INTEGER PRIMARY KEY, incident_key TEXT, old_status TEXT, new_status TEXT, note TEXT, updated_at TEXT)')
    return con


def import_incidents(run_id, events, path=None):
    now=datetime.now(timezone.utc).isoformat()
    with incident_connection(path) as con:
        for e in events.to_dict('records'):
            key=f"{run_id}:{e['event_id']}"
            cur=con.execute('INSERT OR IGNORE INTO incidents VALUES (?,?,?,?,?,?,?,?,?,?)',
                (key,run_id,e['event_id'],e['mission_id'],float(e['start_s']),float(e['confirmed_s']),float(e['end_s']),'待复核','',now))
            if cur.rowcount:
                con.execute('INSERT INTO incident_history(incident_key,old_status,new_status,note,updated_at) VALUES (?,?,?,?,?)',
                            (key,None,'待复核','由确认后的异常事件建立',now))


def update_incident(key,status,note,path=None):
    if status not in ['待复核','已确认','误报','维修中','待复检','已关闭']: raise ValueError('Invalid incident status')
    if not note.strip(): raise ValueError('请填写复核、维修或复检依据')
    allowed={'待复核':{'已确认','误报'},'已确认':{'维修中','待复检','误报'},'维修中':{'待复检'},'待复检':{'已关闭','维修中'},'误报':{'待复核'},'已关闭':{'待复核'}}
    with incident_connection(path) as con:
        row=con.execute('SELECT status FROM incidents WHERE incident_key=?',(key,)).fetchone()
        if row is None: raise ValueError('Incident not found')
        old=row[0]
        if status != old and status not in allowed[old]: raise ValueError(f'不能从 {old} 直接变为 {status}')
        now=datetime.now(timezone.utc).isoformat()
        con.execute('UPDATE incidents SET status=?,note=?,updated_at=? WHERE incident_key=?',(status,note,now,key))
        con.execute('INSERT INTO incident_history(incident_key,old_status,new_status,note,updated_at) VALUES (?,?,?,?,?)',(key,old,status,note,now))


def read_incidents(path=None):
    with incident_connection(path) as con: return pd.read_sql_query('SELECT * FROM incidents ORDER BY updated_at DESC',con)


def incident_history(key,path=None):
    with incident_connection(path) as con:
        return pd.read_sql_query('SELECT * FROM incident_history WHERE incident_key=? ORDER BY id',con,params=(key,))
