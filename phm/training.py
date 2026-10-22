"""Offline experiments. Test groups never choose hyperparameters or thresholds."""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import time
import uuid
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error
from phm.data import ROOT, read_json, write_json, sha256


def new_run(task, config, root=None):
    folder = Path(root or ROOT/'runs') / f'{task}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}'
    folder.mkdir(parents=True)
    write_json(folder/'status.json', {'status': 'running', 'task': task})
    write_json(folder/'config.json', {**config, 'python': platform.python_version(), 'sklearn': sklearn.__version__,
        'created_at': datetime.now(timezone.utc).isoformat(), 'code_sha256': sha256(Path(__file__)),
        'adapter_sha256': sha256(ROOT/'phm/data.py')})
    return folder


def mission_split(w):
    missions = w[['mission_id', 'label']].drop_duplicates()
    if missions.mission_id.duplicated().any(): raise ValueError('Mission has conflicting labels')
    mapping = {}
    for label, group in missions.groupby('label'):
        ids = sorted(group.mission_id)
        if len(ids) < 6: raise ValueError(f'Class {label}: need >=6 independent missions')
        n_test = max(1, int(np.ceil(len(ids)*.2)))
        n_val = max(1, int(np.ceil(len(ids)*.2)))
        for i, mission in enumerate(ids):
            mapping[mission] = 'train' if i < len(ids)-n_test-n_val else ('validation' if i < len(ids)-n_test else 'test')
    return w.mission_id.map(mapping), mapping


def classification(actual, predicted, labels):
    report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    return {'macro_f1': float(f1_score(actual, predicted, labels=labels, average='macro', zero_division=0)),
            'accuracy': float(np.mean(np.asarray(actual)==np.asarray(predicted))), 'report': report,
            'labels': list(labels), 'confusion_matrix': confusion_matrix(actual, predicted, labels=labels).tolist()}


def event_stats(df, flag_col, stride_s=1.):
    """Count contiguous flagged evaluation windows as one event; exposure uses stride."""
    healthy = df[df.label == 0]
    events = 0
    for _, g in healthy.groupby('mission_id'):
        g = g.sort_values('timestamp_s'); flag = g[flag_col].astype(bool)
        starts = flag & (~flag.shift(fill_value=False) | (g.timestamp_s.diff() > stride_s*1.5))
        events += int(starts.sum())
    hours = len(healthy)*stride_s/3600
    faulty = df[df.label != 0]
    recalls = faulty.groupby('mission_id')[flag_col].any()
    return {'false_alarm_events': events, 'healthy_exposure_hours': hours,
        'false_alarm_events_per_hour': events/hours if hours else None,
        'fault_mission_recall': float(recalls.mean()) if len(recalls) else None,
        'detection_delay_s': None, 'delay_reason': 'Dataset provides mission-level damage labels, no onset timestamp.'}


def train_uav(processed=None, run_root=None, trees=160):
    processed = Path(processed or ROOT/'data/processed/uav')
    w = pd.read_parquet(processed/'windows.parquet')
    config = {'trees': trees, 'seed': 42, 'validation_healthy_quantile': .95,
              'data_sha256': sha256(processed/'windows.parquet'), 'split': 'chronological missions within each class, approximately 60/20/20'}
    run = new_run('uav', config, run_root); start = time.monotonic()
    try:
        w['split'], mapping = mission_split(w)
        features = [c for c in w if c.startswith(('acc_', 'gyro_'))]
        if not np.isfinite(w[features]).all().all(): raise ValueError('Non-finite model features')
        tr, va, te = [w[w.split == role] for role in ['train','validation','test']]
        healthy_train = tr[tr.label == 0]; healthy_val = va[va.label == 0]
        if min(len(healthy_train), len(healthy_val)) < 10: raise ValueError('Insufficient healthy windows')
        model = RandomForestClassifier(n_estimators=trees, min_samples_leaf=3, class_weight='balanced', random_state=42, n_jobs=2)
        model.fit(tr[features], tr.label)
        anomaly = IsolationForest(n_estimators=120, random_state=42, n_jobs=2).fit(healthy_train[features])
        threshold = float(np.quantile(-anomaly.score_samples(healthy_val[features]), .95))
        rule_threshold = float(healthy_val.acc_norm_std.quantile(.95))
        # Display-only HI based on healthy training deviation, never called calibrated health.
        center = float(healthy_train.acc_norm_std.median())
        scale = max(float(healthy_train.acc_norm_std.quantile(.95)-center), 1e-6)
        w['predicted_label'] = model.predict(w[features])
        w['class_probability'] = model.predict_proba(w[features]).max(axis=1)
        w['anomaly_score'] = -anomaly.score_samples(w[features])
        w['anomaly_flag'] = w.anomaly_score > threshold
        w['rule_flag'] = w.acc_norm_std > rule_threshold
        w['propulsion_hi_proxy'] = (100-25*np.maximum(0, (w.acc_norm_std-center)/scale)).clip(0,100)
        w['recommended_action'] = np.where(w.predicted_label != 0, '建议检查螺旋桨、紧固件和振动；由人工复核', '当前窗口未识别到损伤；继续监测')
        w['risk_level'] = np.where(w.predicted_label != 0, '需复核', '未检出')
        labels = sorted(int(v) for v in w.label.unique())
        results = {role: classification(g.label, g.predicted_label, labels) for role,g in w.groupby('split')}
        test = w[w.split == 'test']
        results['binary_rule'] = classification((test.label != 0).astype(int), test.rule_flag.astype(int), [0,1])
        results['binary_rf'] = classification((test.label != 0).astype(int), (test.predicted_label != 0).astype(int), [0,1])
        results['anomaly_events'] = event_stats(test, 'anomaly_flag')
        per_mission = [{'mission_id': mid, 'label': int(g.label.iloc[0]),
                        'accuracy': float((g.label == g.predicted_label).mean()), 'windows': len(g)} for mid,g in test.groupby('mission_id')]
        artifact = {'classifier': model, 'anomaly': anomaly, 'features': features, 'anomaly_threshold': threshold,
                    'rule_threshold': rule_threshold, 'health_center': center, 'health_scale': scale}
        joblib.dump(artifact, run/'models.joblib')
        w.to_parquet(run/'predictions.parquet', index=False)
        if (processed/'telemetry.parquet').exists():
            shutil.copy2(processed/'telemetry.parquet', run/'telemetry.parquet')
        pd.DataFrame(per_mission).to_csv(run/'per_mission.csv', index=False)
        pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).to_csv(run/'feature_importance.csv', index=False)
        write_json(run/'split.json', mapping)
        write_json(run/'data_quality.json', read_json(processed/'quality.json'))
        metrics = {'task': 'uav', 'run_id': run.name, 'data_kind': 'real_flight', 'dataset': 'UAV Realistic Fault Dataset',
            'missions': len(mapping), 'windows': len(w), 'duration_s': time.monotonic()-start,
            'split_counts': w[['mission_id','split']].drop_duplicates().split.value_counts().to_dict(),
            'thresholds': {'anomaly': threshold, 'acc_norm_std': rule_threshold}, 'results': results,
            'limitations': ['Mission-level indoor propeller damage classification only.', 'Class probability is uncalibrated; HI is an illustrative vibration deviation proxy.',
                'No onset labels: detection delay unavailable. No cross-airframe validation.', 'Test scores are window-level; overlapping windows are correlated.']}
        write_json(run/'metrics.json', metrics); write_json(run/'status.json', {'status':'complete'})
        return run
    except Exception as e:
        write_json(run/'status.json', {'status':'failed', 'error': str(e)}); raise


def battery_features(curves, threshold=1.4):
    rows, states = [], []
    for bid, g in curves.groupby('battery_id'):
        g = g.sort_values('cycle_idx').copy()
        hit = g[g.capacity_ah <= threshold]
        eol = int(hit.cycle_idx.iloc[0]) if len(hit) else None
        states.append({'battery_id': bid, 'cycles': len(g), 'observed_eol': eol, 'censored': eol is None})
        for i in range(9, len(g)):
            current = g.iloc[i]; cycle = int(current.cycle_idx)
            if eol is None or cycle >= eol: continue
            prefix = g.iloc[max(0,i-19):i+1]
            slope, intercept = np.polyfit(prefix.cycle_idx, prefix.capacity_ah, 1)
            linear = max(0., (threshold-float(current.capacity_ah))/slope) if slope < -1e-8 else np.nan
            rows.append({'battery_id': bid, 'cycle_idx': cycle, 'soh': float(current.soh), 'capacity_ah': float(current.capacity_ah),
                'slope': float(slope), 'recent_mean': float(prefix.capacity_ah.mean()), 'true_rul': eol-cycle, 'linear_rul': linear})
    return pd.DataFrame(rows), states

BATTERY_FEATURES = ['cycle_idx','soh','slope','recent_mean']

def regressor(trees=120): return RandomForestRegressor(n_estimators=trees, min_samples_leaf=4, random_state=42, n_jobs=2)

def regression_metrics(df, column):
    valid = df.dropna(subset=[column])
    if valid.empty: return {'mae': None, 'rmse': None, 'n': 0}
    return {'mae': float(mean_absolute_error(valid.true_rul, valid[column])),
            'rmse': float(np.sqrt(mean_squared_error(valid.true_rul, valid[column]))), 'n': len(valid)}


def train_battery(processed=None, run_root=None, trees=120):
    processed = Path(processed or ROOT/'data/processed/battery')
    curves = pd.read_parquet(processed/'curves.parquet')
    config = {'trees': trees, 'seed':42, 'eol_capacity_ah':1.4, 'rated_capacity_ah':2., 'min_prefix':10,
              'data_sha256': sha256(processed/'curves.parquet'), 'split':'leave-one-battery-out, pre-EOL only'}
    run = new_run('battery', config, run_root); start = time.monotonic()
    try:
        data, states = battery_features(curves)
        if data.empty or data.battery_id.nunique() < 3: raise ValueError('Need at least three specimens with observed EOL. SOH curves remain available.')
        predictions, folds = [], []
        for held in sorted(data.battery_id.unique()):
            tr = data[data.battery_id != held]; te = data[data.battery_id == held].copy()
            model = regressor(trees).fit(tr[BATTERY_FEATURES], tr.true_rul)
            te['predicted_rul'] = model.predict(te[BATTERY_FEATURES]).clip(0)
            # Nested specimen-held-out residuals use training cells only.
            residuals = []
            for cal in sorted(tr.battery_id.unique()):
                fit, val = tr[tr.battery_id != cal], tr[tr.battery_id == cal]
                inner = regressor(trees).fit(fit[BATTERY_FEATURES], fit.true_rul)
                residuals.extend(np.abs(val.true_rul-inner.predict(val[BATTERY_FEATURES])).tolist())
            radius = float(np.quantile(residuals, .9, method='higher'))
            te['lower'] = (te.predicted_rul-radius).clip(lower=0); te['upper'] = te.predicted_rul+radius
            predictions.append(te)
            folds.append({'test_battery': held, 'train_batteries': sorted(tr.battery_id.unique().tolist()), 'interval_radius': radius})
            joblib.dump({'model':model, 'features': BATTERY_FEATURES, 'interval_radius': radius}, run/f'{held}.joblib')
        p = pd.concat(predictions, ignore_index=True)
        p['stage'] = pd.cut(p.cycle_idx/(p.cycle_idx+p.true_rul), [0,1/3,2/3,1.01], labels=['early','middle','late']).astype(str)
        p.to_parquet(run/'predictions.parquet', index=False); curves.to_parquet(run/'curves.parquet', index=False)
        per = {bid: regression_metrics(g, 'predicted_rul') for bid,g in p.groupby('battery_id')}
        metrics = {'task':'battery', 'run_id':run.name, 'data_kind':'real_lab', 'dataset':'NASA Battery', 'duration_s':time.monotonic()-start,
            'eol_capacity_ah':1.4, 'rated_capacity_ah':2., 'states':states, 'folds':folds,
            'random_forest':regression_metrics(p,'predicted_rul'), 'linear':regression_metrics(p,'linear_rul'),
            'common_support': {'random_forest': regression_metrics(p.dropna(subset=['linear_rul']), 'predicted_rul'),
                               'linear': regression_metrics(p.dropna(subset=['linear_rul']), 'linear_rul')},
            'macro_battery_mae': float(np.mean([v['mae'] for v in per.values()])), 'per_battery':per,
            'per_stage': {stage:regression_metrics(g,'predicted_rul') for stage,g in p.groupby('stage')},
            'interval': {'nominal_coverage':.9, 'empirical_coverage':float(((p.true_rul>=p.lower)&(p.true_rul<=p.upper)).mean()),
                'mean_width':float((p.upper-p.lower).mean()), 'status':'exploratory_small_sample',
                'method':'90th percentile absolute nested leave-one-training-cell-out residual; no held-out test target used.'},
            'limitations':['Only pre-EOL discharge cycles scored; censored cells excluded from RUL metrics but retained in SOH views.',
                'SOH = measured capacity / 2 Ah, not model-estimated SOH.', 'Three/few independent EOL cells: interval has no guaranteed coverage.',
                'RUL unit is discharge cycles, not flight hours. Laboratory operating conditions differ between cells.']}
        write_json(run/'metrics.json', metrics); write_json(run/'data_quality.json', read_json(processed/'quality.json'))
        write_json(run/'status.json', {'status':'complete'}); return run
    except Exception as e:
        write_json(run/'status.json', {'status':'failed','error':str(e)}); raise


def list_runs(root=None):
    root = Path(root or ROOT/'runs')
    return sorted([p.parent for p in root.glob('*/metrics.json') if (p.parent/'status.json').exists()
        and read_json(p.parent/'status.json')['status']=='complete'], reverse=True)

if __name__ == '__main__':
    import argparse
    from phm.data import prepare_uav, prepare_battery
    p = argparse.ArgumentParser(); p.add_argument('--task', choices=['all','uav','battery'], default='all'); p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    for task, prep, train in [('uav',prepare_uav,train_uav),('battery',prepare_battery,train_battery)]:
        if args.task in ('all',task):
            if args.prepare: print('Prepared', prep(), flush=True)
            print('Completed', train(), flush=True)
