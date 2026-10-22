"""Business invariants using isolated ledgers and the published local models."""
import json
import pytest
from phm import platform as p

@pytest.fixture
def ledger(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'DB_PATH',tmp_path/'platform.sqlite')
    monkeypatch.setattr(p,'ARTIFACTS_ROOT',tmp_path/'analyses')
    p.init();p.sync_models()
    return p

@pytest.fixture
def demo(ledger):
    if not p.records('SELECT * FROM models'):pytest.skip('Local trained models required')
    p.seed_demo()
    return p


def test_equipment_constraints_and_unknown_health(ledger):
    aid=p.create_asset('实车','UGV','CAR-001')
    with pytest.raises(ValueError):p.create_asset('重复','UAV','CAR-001')
    analysis=p.analyze_asset(aid)
    assert all(r['score'] is None and r['status']=='未评估' for r in p.analysis_results(analysis))
    assert p.latest_analysis(aid)['summary']['health_score'] is None
    assert p.fleet()[0]['coverage']==0
    other=p.create_asset('其他','UAV','AIR-001')
    parent=next(n for n in p.structure(other) if n['kind']=='subsystem')
    with pytest.raises(ValueError):p.add_component(aid,parent['id'],'错误归属')


def test_demo_bindings_and_provenance(demo):
    assets=p.fleet();assert len(assets)==3
    p.seed_demo();assert len(p.fleet())==3
    u=next(a for a in assets if a['kind']=='UAV');b=next(a for a in assets if a['kind']=='部件台架')
    latest=p.latest_analysis(u['id']);assert latest['summary']['coverage']==.2
    result=next(r for r in p.analysis_results(latest['id']) if r['family']=='uav')
    assert result['provenance']['input_sha256'] and result['provenance']['model_sha256']
    battery=p.analysis_results(p.latest_analysis(b['id'])['id'])[0]
    assert battery['cycle']==60 and battery['rul'] is not None
    assert battery['lower']<=battery['rul']<=battery['upper']
    n=next(n for n in p.structure(u['id']) if n['model_id'])
    wrong=next(m for m in p.records('SELECT * FROM models') if m['family']=='battery')
    with pytest.raises(ValueError):p.bind(n['id'],wrong['run_id'],n['source_kind'],n['source_ref'],n['case_id'])
    actual=p.create_asset('实机','UAV','ACTUAL-1')
    c=next(n for n in p.structure(actual) if n['family']=='uav' and n['kind']=='component')
    with pytest.raises(ValueError,match='演示档案'):p.bind(c['id'],n['model_id'],n['source_kind'],n['source_ref'],n['case_id'])
    p.unbind(n['id']);assert p.latest_analysis(u['id'])['stale']
    assert next(a for a in p.fleet() if a['id']==u['id'])['health_status']=='待重新分析'


def test_alert_order_round_trip_and_deduplication(demo):
    alert=p.records('SELECT * FROM alerts')[0];aid=alert['asset_id']
    p.review_alert(alert['id'],'已确认','检查信号证据')
    p.analyze_asset(aid)
    assert len(p.records('SELECT * FROM alerts'))==1
    assert json.loads(p.records('SELECT * FROM alerts')[0]['evidence'])['reviews'][0]['note']=='检查信号证据'
    oid=p.create_order(aid,'检查动力组件','中','演示员',alert_id=alert['id'])
    assert p.create_order(aid,'重复请求','中',alert_id=alert['id'])==oid
    with pytest.raises(ValueError):p.update_order(oid,'已关闭','跳过维修')
    p.update_order(oid,'处理中','开始检查');p.update_order(oid,'待复检','完成维修')
    with pytest.raises(ValueError):p.update_order(oid,'已关闭','结束')
    p.update_order(oid,'已关闭','结束','人工台架复检通过，仅测试流程')
    assert p.records('SELECT * FROM alerts')[0]['status']=='已关闭'
    assert len(p.export_asset(aid)['order_history'])==4
    p.analyze_asset(aid)
    assert p.records('SELECT * FROM alerts')[0]['status']=='已关闭'


def test_disabled_model_invalidates_latest_analysis(demo):
    n=next(n for a in p.fleet() for n in p.structure(a['id']) if n['model_id'])
    p.configure_model(n['model_id'],'暂停模型',False,'待验证')
    assert p.latest_analysis(n['asset_id'])['stale']
    an=p.analyze_asset(n['asset_id'])
    assert next(r for r in p.analysis_results(an) if r['node_id']==n['id'])['status']=='无法评估'
