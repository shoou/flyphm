"""Causal confirmation/release plus retrospective event grouping.
Operating regimes are angular-motion proxies, NOT verified takeoff/landing phases.
"""
import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

EVENT_COLUMNS = ['event_id','mission_id','start_s','confirmed_s','end_s','duration_s','active_seconds','peak_score','predicted_label','episodes','end_reason']


def condition_regimes(w, boundary):
    return np.where(w.gyro_norm_rms <= boundary, 'low_angular_motion', 'high_angular_motion')


def fit_conditioned(train, validation, features, quantile=.975):
    healthy = train[train.label == 0]
    val = validation[validation.label == 0]
    boundary = float(healthy.gyro_norm_rms.median())
    models = {}
    fallback = IsolationForest(n_estimators=120, random_state=42, n_jobs=2).fit(healthy[features])
    global_threshold = float(np.quantile(-fallback.score_samples(val[features]), quantile))
    for regime in ['low_angular_motion','high_angular_motion']:
        tr = healthy[condition_regimes(healthy,boundary)==regime]
        va = val[condition_regimes(val,boundary)==regime]
        if len(tr) >= 40 and len(va) >= 20:
            model = IsolationForest(n_estimators=120, random_state=42, n_jobs=2).fit(tr[features])
            threshold = float(np.quantile(-model.score_samples(va[features]), quantile))
            models[regime] = {'model':model,'threshold':threshold,'train_healthy_windows':len(tr),'validation_healthy_windows':len(va),'fallback':False}
        else:
            models[regime] = {'model':fallback,'threshold':global_threshold,'train_healthy_windows':len(tr),'validation_healthy_windows':len(va),'fallback':True}
    return {'boundary':boundary,'regimes':models,'features':features,'quantile':quantile}


def score_conditioned(w, bundle):
    result = w.copy()
    result['operating_regime'] = condition_regimes(result,bundle['boundary'])
    result['anomaly_score'] = np.nan; result['anomaly_threshold'] = np.nan
    for regime, params in bundle['regimes'].items():
        mask = result.operating_regime == regime
        if mask.any():
            result.loc[mask,'anomaly_score'] = -params['model'].score_samples(result.loc[mask,bundle['features']])
            result.loc[mask,'anomaly_threshold'] = params['threshold']
    result['raw_anomaly_flag'] = result.anomaly_score > result.anomaly_threshold
    return result


def confirm_alerts(w, confirm=3, release=2, merge_gap_s=3., stride_s=1.):
    """Output causal active flags. Merge event records only, never rewrite past flags."""
    if confirm < 1 or release < 1 or merge_gap_s < 0: raise ValueError('Invalid alarm parameters')
    result = w.copy(); result['confirmed_alert'] = False
    episodes = []
    for mission, g in result.groupby('mission_id',sort=False):
        g = g.sort_values('timestamp_s')
        active = False; on_count = off_count = 0; pending = None; current = None; previous = None; segment = 0
        for idx,row in g.iterrows():
            t = float(row.timestamp_s)
            valid = bool(np.isfinite(row.anomaly_score) and np.isfinite(row.anomaly_threshold))
            broken = previous is not None and (t-previous > stride_s*1.5 or t <= previous)
            if broken or not valid:
                if current:
                    current['end_reason']='data_gap'; episodes.append(current); current=None
                active=False; on_count=off_count=0; pending=None; segment+=1
            if not valid:
                previous=t; continue
            raw = bool(row.raw_anomaly_flag)
            if not active:
                if raw:
                    if on_count==0: pending=t
                    on_count+=1
                    if on_count>=confirm:
                        active=True; off_count=0
                        current={'mission_id':str(mission),'start_s':pending,'confirmed_s':t,'end_s':t,
                                 'active_seconds':0.,'peak_score':float(row.anomaly_score),
                                 'predicted_label':int(row.get('predicted_label',-1)), 'episodes':1,'segment':segment,'end_reason':'recording_end'}
                else: on_count=0; pending=None
            else:
                off_count = 0 if raw else off_count+1
                if off_count>=release:
                    current['end_reason']='cleared'; episodes.append(current); current=None
                    active=False; on_count=off_count=0; pending=None
            result.loc[idx,'confirmed_alert']=active
            if active:
                current['end_s']=t; current['active_seconds']+=stride_s
                if row.anomaly_score>current['peak_score']:
                    current['peak_score']=float(row.anomaly_score)
                    current['predicted_label']=int(row.get('predicted_label',-1))
            previous=t
        if current: episodes.append(current)
    merged=[]
    for e in episodes:
        if merged and e['mission_id']==merged[-1]['mission_id'] and e['segment']==merged[-1]['segment'] and e['confirmed_s']-merged[-1]['end_s']-stride_s<=merge_gap_s:
            prev=merged[-1]; prev['end_s']=e['end_s']; prev['active_seconds']+=e['active_seconds']; prev['episodes']+=1; prev['end_reason']=e['end_reason']
            if e['peak_score']>prev['peak_score']: prev['peak_score']=e['peak_score']; prev['predicted_label']=e['predicted_label']
        else: merged.append(e.copy())
    for e in merged:
        e['duration_s']=e['end_s']-e['start_s']+stride_s
        e['event_id']=hashlib.sha256(f"{e['mission_id']}:{e['start_s']}:{e['confirmed_s']}".encode()).hexdigest()[:16]
    return result,pd.DataFrame(merged,columns=EVENT_COLUMNS)


def alarm_metrics(w, flag='confirmed_alert', events=None):
    healthy=w[w.label==0]; faulty=w[w.label!=0]
    if events is None:
        count=0
        for _,g in healthy.groupby('mission_id'):
            g=g.sort_values('timestamp_s'); f=g[flag].astype(bool)
            count+=int((f & (~f.shift(fill_value=False)|(g.timestamp_s.diff()>1.5))).sum())
    else: count=int(events.mission_id.isin(healthy.mission_id.unique()).sum())
    hours=len(healthy)/3600
    return {'false_alarm_events':count,'healthy_exposure_hours':hours,
            'false_alarm_events_per_hour':count/hours if hours else None,
            'false_alarm_seconds':int(healthy[flag].sum()),
            'healthy_alarm_fraction':float(healthy[flag].mean()) if len(healthy) else None,
            'fault_window_recall':float(faulty[flag].mean()) if len(faulty) else None,
            'fault_mission_recall':float(faulty.groupby('mission_id')[flag].any().mean()) if len(faulty) else None,
            'detection_delay_s':None,'delay_reason':'No fault-onset labels; confirmation time is not detection latency.'}
