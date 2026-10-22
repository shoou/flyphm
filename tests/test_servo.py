"""Servo diagnosis quality, causal evidence and business integration regressions."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from phm import servo
from phm import platform as p
from phm.data import read_json
from phm.training import list_runs


def test_vibration_features_and_invalid_windows():
    rate=44100;t=np.arange(rate)/rate
    f=servo.signal_features(.2*np.sin(2*np.pi*1000*t),rate)
    assert f['rms']==pytest.approx(.2/np.sqrt(2))
    assert f['dominant_hz']==pytest.approx(1000,abs=15)
    assert f['band_500_2000']>.99
    for x in [np.zeros(rate),np.full(rate,np.nan),np.ones(rate-1)]:
        with pytest.raises(ValueError):servo.signal_features(x,rate)
    assert not {'label','speed_pct','reversal','condition','record_id'}&set(servo.FEATURES)


def test_servo_confirmation_is_causal_and_resets_on_gaps():
    w=pd.DataFrame({'timestamp_s':[1.,2.,4.,5.,6.,7.,8.,9.,10.],
                    'predicted_label':[1,1,1,1,1,0,2,2,2],'class_probability':[.8]*9})
    assert not servo.fault_events(w[w.timestamp_s<=5])
    events=servo.fault_events(w)
    assert [e['evidence']['confirmed_s'] for e in events]==[6.,10.]
    assert events[0]['evidence']['start_s']==4
    assert events[0]['evidence']['end_s']==6
    assert events[0]['key']==servo.fault_events(w[w.timestamp_s<=6])[0]['key']


@pytest.fixture
def run():
    runs=[r for r in list_runs() if r.name.startswith('servo_')]
    if not runs:pytest.skip('Real servo artifacts not downloaded/trained yet')
    return runs[0]


def test_real_recordings_are_speed_isolated_and_reports_reproducible(run):
    w=pd.read_parquet(run/'predictions.parquet');m=read_json(run/'metrics.json')
    assert w.groupby('record_id').split.nunique().max()==1
    assert w.groupby('speed_pct').split.nunique().max()==1
    assert set(w[w.split=='test'].speed_pct)=={20,50}
    assert set(w[w.split=='validation'].speed_pct)=={10,40}
    assert set(w[w.split=='test'].label)==set(servo.LABELS)
    assert w.timestamp_s.min()>=2  # excluded startup, one second windows end at 2 s
    assert m['selected_model']==max(m['selection'],key=lambda x:x['validation_record_macro_f1'])['model']
    assert (run/'validation_report.html').exists()
    rp=servo.record_predictions(w);actual=pd.read_csv(run/'record_predictions.csv')
    pd.testing.assert_frame_equal(rp,actual)
    assert m['rul'] is None and m['detection_delay'] is None


def test_servo_predictions_ignore_source_labels_and_honor_prefix(run,tmp_path):
    w=pd.read_parquet(run/'predictions.parquet');rid=w.record_id.iloc[0]
    expected=servo.score_case(run,run,rid,5)
    w['label']=(w.label+1)%4;w.to_parquet(tmp_path/'predictions.parquet',index=False)
    actual=servo.score_case(run,tmp_path,rid,5)
    assert actual.timestamp_s.max()<=5
    assert expected.predicted_label.tolist()==actual.predicted_label.tolist()
    assert expected.class_probability.tolist()==actual.class_probability.tolist()


def test_servo_asset_to_order_round_trip(run,tmp_path,monkeypatch):
    monkeypatch.setattr(p,'DB_PATH',tmp_path/'business.sqlite')
    monkeypatch.setattr(p,'ARTIFACTS_ROOT',tmp_path/'analyses')
    aid=p.seed_servo_demo();assert p.seed_servo_demo()==aid
    latest=p.latest_analysis(aid);results=p.analysis_results(latest['id'])
    assert results[0]['family']=='servo' and results[0]['rul'] is None and results[0]['score'] is None
    assert results[0]['provenance']['labels_used_for_inference'] is False
    node=next(n for n in p.structure(aid) if n['model_id'])
    with pytest.raises(ValueError,match='类型不匹配'):
        p.bind(node['id'],run.name,'uav_case',run.name,node['case_id'])
    alerts=p.records('SELECT * FROM alerts WHERE asset_id=?',(aid,))
    if alerts:
        oid=p.create_order(aid,'伺服台架检查','中',alert_id=alerts[0]['id'])
        p.update_order(oid,'处理中','人工确认紧固与振动')
        p.update_order(oid,'待复检','完成测试维护流程')
        p.update_order(oid,'已关闭','完成','隔离测试复检记录')
        assert p.records('SELECT status FROM orders WHERE id=?',(oid,))[0]['status']=='已关闭'
    p.configure_model(run.name,'停用伺服模型',False,'测试')
    assert p.latest_analysis(aid)['stale']
    an=p.analyze_asset(aid)
    assert p.analysis_results(an)[0]['status']=='无法评估'


def test_prepare_rejects_tampering_and_marks_bad_windows(tmp_path):
    from scipy.io import wavfile
    from phm.data import write_json,sha256
    raw=tmp_path/'raw';raw.mkdir();records=[]
    rate=44100;t=np.arange(rate*4)/rate
    for label in range(4):
        for speed in [5,10,20]:
            rid=f'fixture_{label}_{speed}';path=raw/(rid+'.wav')
            signal=.2*np.sin(2*np.pi*(500+label*500+speed)*t)
            signal[:rate]=0  # documented startup removal must not be evaluated
            if label==0 and speed==5:signal[2*rate:3*rate]=0
            wavfile.write(path,rate,(signal*32767).astype(np.int16))
            records.append({'record_id':rid,'condition':'test_fixture','label':label,'speed_pct':speed,'reversal':False,'path':path.name,'sha256':sha256(path)})
    write_json(raw/'manifest.json',{'version':13,'sensor':'vibrometer_wav','recordings':records})
    prepared=servo.prepare(raw,tmp_path/'processed')
    w=pd.read_parquet(prepared/'windows.parquet')
    assert w.timestamp_s.min()==2
    assert not ((w.record_id=='fixture_0_5')&(w.timestamp_s==3)).any()
    q=read_json(prepared/'quality.json')['records'][0]
    assert q['trimmed_start_s']==1 and q['unknown_windows'][0]['second']==3
    (raw/records[0]['path']).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='校验失败'):servo.prepare(raw,tmp_path/'bad')
