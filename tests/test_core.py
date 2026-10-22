import numpy as np
import pandas as pd
import pytest
from source.phm_demo import health_scores, rul_from_history, make_windows, make_data, train, predict, FEATURES
from phm.data import signal_windows, CHANNELS
from phm.training import mission_split, battery_features, event_stats
from phm.events import save_event, read_events


def test_missing_health_is_unknown_not_full_score():
    result = health_scores({k:np.nan for k in FEATURES})
    assert result['overall'] is None
    assert all(x is None for x in result['subsystems'].values())
    assert result['score_status'] == 'unknown_missing_data'


def test_rul_seconds_convert_to_hours():
    h = pd.DataFrame({'timestamp_s':np.arange(12)*3600., 'vibration_rms':np.arange(12)*.1})
    r = rul_from_history(h, threshold=1.2)
    assert r['rul_hours'] == pytest.approx(1.)
    assert r['slope_per_hour'] == pytest.approx(.1)


def test_unlabeled_inference_windows_do_not_cross_missions():
    d = pd.DataFrame({k:np.r_[np.ones(12),np.ones(12)*9] for k in FEATURES})
    d['device_id']='same'; d['device_type']='UAV'
    d['mission_id']=['a']*12+['b']*12; d['timestamp_s']=list(range(12))*2
    w = make_windows(d)
    assert len(w)==2 and 'fault_label' not in w
    assert set(w.motor_current_a)=={1.,9.}


def test_mission_split_class_coverage_and_no_overlap():
    w = pd.DataFrame([{'mission_id':f'{label}_{i:02}', 'label':label} for label in range(5) for i in range(8) for _ in range(3)])
    roles, mapping = mission_split(w)
    assert w.assign(role=roles).groupby('mission_id').role.nunique().max()==1
    for role in ['train','validation','test']:
        assert set(w[roles==role].label)==set(range(5))
    assert mapping['0_07']=='test' and mapping['0_00']=='train'


def test_windows_reject_sensor_gap():
    t = np.r_[np.arange(0,.5,.01),np.arange(1,2.1,.01)]
    d = pd.DataFrame({k:np.ones(len(t)) for k in CHANNELS}); d['timestamp_s']=t
    assert signal_windows(d,'m').empty


def test_battery_prefix_is_causal_and_post_eol_excluded():
    d = pd.DataFrame({'battery_id':'B','cycle_idx':np.arange(1,51), 'capacity_ah':2-np.arange(50)*.02})
    d['soh']=d.capacity_ah/2
    a, states = battery_features(d)
    modified = d.copy(); modified.loc[modified.cycle_idx>20,'capacity_ah']-=.1
    modified['soh']=modified.capacity_ah/2
    b, _ = battery_features(modified)
    cols = ['soh','slope','recent_mean','linear_rul']
    np.testing.assert_allclose(a[a.cycle_idx==20][cols],b[b.cycle_idx==20][cols])
    assert (a.true_rul>0).all()
    assert a.cycle_idx.max()<states[0]['observed_eol']


def test_censored_battery_retained_in_state_not_rul():
    d = pd.DataFrame({'battery_id':'B','cycle_idx':range(1,31),'capacity_ah':np.linspace(2,1.8,30)})
    d['soh']=d.capacity_ah/2
    rows, states = battery_features(d)
    assert rows.empty and states[0]['censored']


def test_contiguous_alerts_merge_but_gaps_split():
    d = pd.DataFrame({'mission_id':'m','timestamp_s':[2,3,4,8,9], 'label':0,'flag':[True,True,False,True,True]})
    stats = event_stats(d,'flag')
    assert stats['false_alarm_events']==2
    assert stats['healthy_exposure_hours']==pytest.approx(5/3600)
    assert stats['detection_delay_s'] is None


def test_review_ledger_updates_without_duplicate(tmp_path):
    db=tmp_path/'events.sqlite'
    save_event('r','m',1,2,'待复核','check',db)
    save_event('r','m',1,2,'误报','reviewed',db)
    rows=read_events(db)
    assert len(rows)==1 and rows.iloc[0].status=='误报'


def test_legacy_unlabeled_inference_preserves_unknown(tmp_path):
    data = make_data()
    train(data, tmp_path/'models')
    unlabeled = data.drop(columns='fault_label').copy()
    unlabeled.loc[unlabeled.device_id=='D000', FEATURES] = np.nan
    predict(unlabeled, tmp_path/'models/models.joblib', tmp_path/'pred.csv')
    output = pd.read_csv(tmp_path/'pred.csv')
    missing = output[output.device_id=='D000']
    assert (missing.predicted_fault=='unknown').all()
    assert missing.overall.isna().all()
    assert missing.anomaly_flag.isna().all()
