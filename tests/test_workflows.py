import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from phm.alerts import confirm_alerts, alarm_metrics, score_conditioned
from phm.imports import inspect_csv, save_import, infer_import, REQUIRED
from phm.events import import_incidents, update_incident, read_incidents, incident_history
from phm.experiments import compare_runs
from phm.training import list_runs


def flags(values,times=None,mission='m'):
    return pd.DataFrame({'mission_id':mission,'timestamp_s':times if times is not None else np.arange(len(values)),
        'anomaly_score':np.asarray(values,dtype=float),'anomaly_threshold':.5,'raw_anomaly_flag':np.asarray(values,dtype=bool),'label':0,'predicted_label':0})


def test_confirmation_release_are_causal():
    w=flags([1,0,1,1,1,0,0,1])
    full,events=confirm_alerts(w,3,2,0)
    assert full.confirmed_alert.tolist()==[False,False,False,False,True,True,False,False]
    assert events.iloc[0].start_s==2 and events.iloc[0].confirmed_s==4
    for n in range(1,len(w)+1):
        prefix,_=confirm_alerts(w.iloc[:n],3,2,0)
        assert prefix.confirmed_alert.tolist()==full.confirmed_alert.iloc[:n].tolist()


def test_missing_or_gapped_windows_reset_confirmation_and_prevent_merge():
    w=flags([1,1,1,1,1,1],[0,1,2,20,21,22])
    out,events=confirm_alerts(w,3,2,100)
    assert out.confirmed_alert.tolist()==[False,False,True,False,False,True]
    assert len(events)==2 and events.iloc[0].end_reason=='data_gap'
    w.loc[2,'anomaly_score']=np.nan
    out,events=confirm_alerts(w,3,2,100)
    assert not out.confirmed_alert.iloc[2]


def test_event_merge_does_not_rewrite_active_duration():
    out,events=confirm_alerts(flags([1,1,0,0,1,1,0,0]),1,1,3)
    assert len(events)==1 and events.iloc[0].episodes==2
    assert events.iloc[0].active_seconds==4
    assert alarm_metrics(out,events=events)['false_alarm_seconds']==4
    a=flags([1,1]);b=flags([1,1],mission='other')
    _,separate=confirm_alerts(pd.concat([a,b],ignore_index=True),1,1,100)
    assert len(separate)==2


def csv_data():
    n=600;t=np.arange(n)/100
    frame=pd.DataFrame({'device':'A','flight':'F','clock':t*1000,'ax':np.zeros(n),'ay':np.zeros(n),'az':np.ones(n),
        'gx':np.ones(n)*180,'gy':np.zeros(n),'gz':np.zeros(n)})
    mapping=dict(zip(REQUIRED,['device','flight','clock','ax','ay','az','gx','gy','gz']))
    return frame,mapping,{'time':'ms','acc':'g','gyro':'deg/s'}


def test_import_units_task_identity_and_no_label_required(tmp_path):
    df,mapping,units=csv_data()
    other=df.copy();other.device='B'
    data=pd.concat([df,other],ignore_index=True).to_csv(index=False).encode()
    out,quality=inspect_csv(data,mapping,units)
    assert not quality['blocking_errors']
    assert out.mission_id.nunique()==2 and 'label' not in out
    assert out.acc_z.iloc[0]==pytest.approx(9.80665)
    assert out.gyro_x.iloc[0]==pytest.approx(np.pi)
    assert out.timestamp_s.max()==pytest.approx(5.99)
    saved=save_import(data,mapping,units,root=tmp_path)
    assert (saved/'input.csv').read_bytes()==data


def test_missing_channels_cannot_create_predictions(tmp_path):
    df,mapping,units=csv_data();mapping['acc_x']=None
    out,report=inspect_csv(df.to_csv(index=False).encode(),mapping,units)
    assert report['status']=='partial' and out.acc_x.isna().all()
    from phm.data import signal_windows
    assert signal_windows(out,out.mission_id.iloc[0]).empty


def test_bad_mapping_time_and_duplicates_are_reported():
    df,mapping,units=csv_data()
    mapping['device_id']=None
    _,q=inspect_csv(df.to_csv(index=False).encode(),mapping,units)
    assert q['blocking_errors']
    mapping['device_id']='device';df.loc[1,'clock']=df.loc[0,'clock'];df.loc[2,'clock']=np.inf
    _,q=inspect_csv(df.to_csv(index=False).encode(),mapping,units)
    assert q['invalid_time_rows']==1 and q['missions'][0]['duplicate_rows']==2
    mapping['gyro_x']='ax'
    with pytest.raises(ValueError,match='多个字段'): inspect_csv(df.to_csv(index=False).encode(),mapping,units)


def test_incident_history_cannot_skip_reinspection(tmp_path):
    db=tmp_path/'events.sqlite'
    _,events=confirm_alerts(flags([1,1,1,0,0]),2,2,0)
    import_incidents('r',events,db);import_incidents('r',events,db)
    ledger=read_incidents(db);assert len(ledger)==1
    key=ledger.iloc[0].incident_key
    with pytest.raises(ValueError):update_incident(key,'已关闭','unverified',db)
    for state in ['已确认','维修中','待复检','已关闭']:update_incident(key,state,'有据可查',db)
    assert len(incident_history(key,db))==5
    import_incidents('r',events,db)
    assert read_incidents(db).iloc[0].status=='已关闭'


def test_comparison_refuses_changed_split(tmp_path):
    candidates=[r for r in list_runs() if (r/'alert_policy.joblib').exists()]
    if not candidates: pytest.skip('No real derived experiment')
    import shutil
    right=candidates[0]
    parent=json.loads((right/'config.json').read_text())['parent_run']
    left=right.parent/parent
    assert compare_runs(left,right)['comparable']
    clone=tmp_path/'changed';shutil.copytree(right,clone)
    split=json.loads((clone/'split.json').read_text());split[next(iter(split))]='test'
    (clone/'split.json').write_text(json.dumps(split))
    with pytest.raises(ValueError,match='splits differ'):compare_runs(left,clone)


def test_unlabeled_import_exposes_missing_windows_as_unknown(tmp_path):
    candidates=[r for r in list_runs() if (r/'alert_policy.joblib').exists()]
    if not candidates: pytest.skip('No real model artifact')
    df,mapping,units=csv_data();df.loc[200:250,'ax']=np.nan
    folder=save_import(df.to_csv(index=False).encode(),mapping,units,root=tmp_path)
    output=infer_import(folder,candidates[0])
    status=json.loads((output/'status.json').read_text())
    assert status['metrics_available'] is False
    assert status['missions'][0]['status']=='partial'
    grid=pd.read_parquet(output/'window_quality.parquet')
    assert set(grid.window_status)=={'evaluated','unknown'}
    predictions=pd.read_parquet(output/'predictions.parquet')
    assert not set(predictions.timestamp_s)&set(grid[grid.window_status=='unknown'].timestamp_s)
    mapping['acc_y']=None
    folder=save_import(df.to_csv(index=False).encode(),mapping,units,root=tmp_path)
    output=infer_import(folder,candidates[0])
    assert json.loads((output/'status.json').read_text())['status']=='unknown'
    assert not (output/'predictions.parquet').exists()
