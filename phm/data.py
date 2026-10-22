"""Dataset adapters with explicit provenance, units and quality reporting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def read_tii(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            obj = json.loads(line)
            rows.append([obj['timestamp'] / 1e6, *obj['accelerometer_m_s2'], *obj['gyro_rad'], obj.get('accelerometer_clipping', 0)])
    df = pd.DataFrame(rows, columns=['timestamp_s', *CHANNELS, 'clipping'])
    raw_rows = len(df)
    reversed_steps = int((df.timestamp_s.diff() < 0).sum())
    duplicates = int(df.timestamp_s.duplicated().sum())
    df = df.sort_values('timestamp_s').drop_duplicates('timestamp_s')
    invalid = ~np.isfinite(df[['timestamp_s', *CHANNELS]]).all(axis=1) | (df.clipping != 0)
    invalid_rows = int(invalid.sum()); df = df.loc[~invalid].copy()
    if len(df) < 100: raise ValueError(f'Too few valid samples: {path}')
    df['timestamp_s'] -= df.timestamp_s.iloc[0]
    return df.reset_index(drop=True), {'raw_rows': raw_rows, 'invalid_rows': invalid_rows,
        'duplicates': duplicates, 'out_of_order_steps': reversed_steps, 'valid_rows': len(df),
        'median_dt_s': float(df.timestamp_s.diff().median())}

def signal_windows(df, mission_id, label=None, window_s=2., stride_s=1.):
    """Fixed-duration causal windows. Never cross a mission or a >0.2 s gap."""
    rows = []
    for stop in np.arange(window_s, df.timestamp_s.max() + 1e-9, stride_s):
        w = df[(df.timestamp_s >= stop-window_s) & (df.timestamp_s < stop)]
        if len(w) < 100 or w.timestamp_s.diff().max() > .2: continue
        if not np.isfinite(w[CHANNELS]).all().all(): continue
        row = {'mission_id': mission_id, 'timestamp_s': float(stop), 'start_s': float(stop-window_s), 'samples': len(w)}
        if label is not None: row['label'] = int(label)
        for c in CHANNELS:
            v = w[c].to_numpy()
            for stat, value in [('mean', v.mean()), ('std', v.std()), ('rms', np.sqrt(np.mean(v*v))), ('ptp', np.ptp(v))]:
                row[f'{c}_{stat}'] = float(value)
        row['acc_norm_std'] = float(np.linalg.norm(w[CHANNELS[:3]], axis=1).std())
        row['gyro_norm_rms'] = float(np.sqrt(np.mean(np.sum(w[CHANNELS[3:]].to_numpy()**2, axis=1))))
        rows.append(row)
    return pd.DataFrame(rows)

def prepare_uav(raw_root=None, out=None):
    raw_root = Path(raw_root or ROOT/'data/raw/tii'); out = Path(out or ROOT/'data/processed/uav')
    manifest = read_json(raw_root/'manifest.json'); windows, traces, reports = [], [], []
    for entry in manifest['files']:
        path = raw_root/entry['path']
        if sha256(path) != entry['sha256']: raise ValueError(f'Checksum mismatch: {path.name}')
        mission = path.parent.name
        df, report = read_tii(path)
        w = signal_windows(df, mission, entry['class'])
        if w.empty: raise ValueError(f'No usable windows: {mission}')
        windows.append(w)
        # UI traces are downsampled; model features use all valid native samples.
        trace = df.iloc[::10].copy(); trace['mission_id'] = mission; traces.append(trace)
        reports.append({'mission_id': mission, 'label': entry['class'], 'windows': len(w), **report})
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(windows, ignore_index=True).to_parquet(out/'windows.parquet', index=False)
    pd.concat(traces, ignore_index=True).to_parquet(out/'telemetry.parquet', index=False)
    write_json(out/'quality.json', {'source': manifest['source'], 'revision': manifest['revision'],
        'manifest_sha256': sha256(raw_root/'manifest.json'), 'data_kind': 'real_flight',
        'units': {'timestamp_s': 's (PX4 timestamp us / 1e6)', 'acc': 'm/s²', 'gyro': 'rad/s'},
        'window_s': 2, 'stride_s': 1, 'missions': reports,
        'label_scope': 'Mission-level number of broken propellers; no in-flight onset labels.',
        'limitations': 'Indoor missions. Takeoff/landing retained. Same platform; no cross-airframe claim.'})
    return out

def battery_capacities(path):
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    cells = np.atleast_1d(raw[Path(path).stem].cycle)
    rows = []
    for source_cycle, cycle in enumerate(cells):
        if str(cycle.type).strip().lower() != 'discharge': continue
        cap = float(np.asarray(cycle.data.Capacity).squeeze())
        if not np.isfinite(cap) or cap <= 0: raise ValueError('Invalid discharge capacity')
        rows.append({'source_cycle': source_cycle, 'capacity_ah': cap,
                     'ambient_temperature_c': float(cycle.ambient_temperature)})
    df = pd.DataFrame(rows)
    df['cycle_idx'] = np.arange(1, len(df)+1)
    df['battery_id'] = Path(path).stem
    df['soh'] = df.capacity_ah / 2.0
    return df

def prepare_battery(raw_root=None, out=None):
    raw_root = Path(raw_root or ROOT/'data/raw/nasa'); out = Path(out or ROOT/'data/processed/battery')
    paths = [raw_root/f'{name}.mat' for name in ['B0005','B0006','B0007','B0018']]
    missing = [p.name for p in paths if not p.exists()]
    if missing: raise FileNotFoundError(f'Missing original NASA specimens: {missing}')
    manifest = read_json(raw_root/'manifest.json')
    for path in paths:
        expected = manifest.get('files', {}).get(path.name)
        if expected is None or sha256(path) != expected:
            raise ValueError(f'NASA source checksum missing or mismatched: {path.name}; rerun downloader')
    curves = pd.concat([battery_capacities(p) for p in paths], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True); curves.to_parquet(out/'curves.parquet', index=False)
    write_json(out/'quality.json', {'data_kind': 'real_lab', 'source': read_json(raw_root/'manifest.json')['source'],
        'rated_capacity_ah': 2., 'eol_capacity_ah': 1.4, 'cycle_unit': 'observed discharge cycle',
        'files': [{'file': p.name, 'sha256': sha256(p)} for p in paths],
        'limitations': 'Four laboratory cells; different discharge cutoffs. Not UAV flight-hour RUL.'})
    return out
