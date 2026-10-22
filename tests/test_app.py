"""Exercise real-data pages when local experiment artifacts are available."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from phm.training import list_runs
import phm.events
import phm.platform

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def app(tmp_path, monkeypatch):
    # UI verification never inserts test review events into the user's ledger.
    monkeypatch.setattr(phm.events, 'ROOT', tmp_path)
    monkeypatch.setattr(phm.platform, 'DB_PATH', tmp_path/'platform.sqlite')
    monkeypatch.setattr(phm.platform, 'ARTIFACTS_ROOT', tmp_path/'analyses')
    phm.platform.seed_demo()
    return AppTest.from_file(str(ROOT/'src/app.py'), default_timeout=40).run()


def assert_no_errors(at):
    assert not at.exception, [e.message for e in at.exception]


def test_overview_and_training_page(app):
    assert_no_errors(app)
    navigate(app,'模型中心','模型训练与验证')
    assert_no_errors(app)


def test_battery_censored_and_prefix_navigation(app):
    if not any(r.name.startswith('battery_') for r in list_runs()): pytest.skip('NASA experiment not present')
    navigate(app,'健康分析','电池健康与寿命'); assert_no_errors(app)
    slider = next(s for s in app.slider if s.label=='已观测到的放电循环')
    slider.set_value(20).run(); assert_no_errors(app)
    next(c for c in app.checkbox if '未来' in c.label).check().run(); assert_no_errors(app)
    next(s for s in app.selectbox if s.label=='电池样本').select('B0007').run(); assert_no_errors(app)
    assert next(m.value for m in app.metric if m.label=='预测剩余循环')=='—'


def test_replay_seek_and_review(app):
    if not any(r.name.startswith('uav_') for r in list_runs()): pytest.skip('UAV experiment not present')
    navigate(app,'健康分析','故障诊断回放'); assert_no_errors(app)
    app.slider[0].set_value(10).run(); assert_no_errors(app)
    next(b for b in app.button if b.label=='将当前窗口加入待复核事件').click().run(); assert_no_errors(app)
    next(b for b in app.button if b.label=='播放').click().run(); assert_no_errors(app)
    assert app.session_state['playing']
    next(b for b in app.button if b.label=='暂停').click().run(); assert_no_errors(app)
    assert not app.session_state['playing']
    navigate(app,'维修管理','历史窗口复核'); assert_no_errors(app)
    next(s for s in app.selectbox if s.label=='处理状态').select('已确认')
    next(b for b in app.button if b.label=='保存处理记录').click().run(); assert_no_errors(app)
    assert phm.events.read_events().iloc[0].status=='已确认'


def test_import_preview_and_model_comparison(app):
    if not any((r/'alert_policy.joblib').exists() for r in list_runs()): pytest.skip('No alert experiment')
    navigate(app,'模型中心','模型版本对比');assert_no_errors(app)
    assert any('旧测试' in w.value for w in app.warning)
    navigate(app,'健康分析','数据导入与推理');assert_no_errors(app)
    next(b for b in app.button if b.label=='载入现有真实飞行任务作为导入示例').click().run();assert_no_errors(app)
    assert any(e.label=='保存此数据集' and not e.disabled for e in app.button)
    # Missing axis is allowed to save as partial, but cannot silently claim a healthy prediction.
    next(s for s in app.selectbox if s.label=='acc_x').select('（未提供）').run();assert_no_errors(app)
    assert any('缺少传感器' in c.value for c in app.caption)


def test_incident_ui_lifecycle(app):
    if not any((r/'alert_policy.joblib').exists() for r in list_runs()): pytest.skip('No alert experiment')
    navigate(app,'维修管理','异常事件与复核');assert_no_errors(app)
    next(b for b in app.button if b.label=='将所选完整事件加入复核台账').click().run();assert_no_errors(app)
    next(s for s in app.selectbox if s.label=='新状态').select('已确认')
    next(t for t in app.text_area if t.label=='处理依据 / 维修措施 / 复检结果').set_value('人工核对振动证据')
    next(b for b in app.button if b.label=='保存事件处理').click().run();assert_no_errors(app)
    assert phm.events.read_incidents().iloc[0].status=='已确认'


def navigate(app,main,sub=None):
    from src.navigation import MENUS
    app.button('menu_'+main).click().run()
    if sub:
        route=next(route for route,title in MENUS[main] if title==sub)
        app.button('nav_'+route).click().run()
    return app


def test_five_business_pages_and_analysis(app):
    for page in ['监测中心','装备管理','健康分析','维修管理','模型中心']:
        navigate(app,page);assert_no_errors(app)
    navigate(app,'健康分析')
    next(b for b in app.button if b.label=='运行装备健康分析').click().run()
    assert_no_errors(app)
    assert any(m.label=='有效覆盖' for m in app.metric)


def test_alert_to_maintenance_workflow(app):
    navigate(app,'监测中心','告警与处置')
    next(b for b in app.button if b.label=='生成维修工单').click().run();assert_no_errors(app)
    navigate(app,'维修管理')
    assert any(m.label=='待处理' and m.value=='1' for m in app.metric)
    for status in ['处理中','待复检','已关闭']:
        next(s for s in app.selectbox if s.label=='下一状态').select(status)
        next(t for t in app.text_area if t.label=='处理记录').set_value('测试流程记录')
        if status=='已关闭':next(t for t in app.text_area if '关闭必填' in t.label).set_value('人工复检通过（隔离测试）')
        next(b for b in app.button if b.label=='保存工单进展').click().run();assert_no_errors(app)
    assert phm.platform.records('SELECT * FROM orders')[0]['status']=='已关闭'


def test_every_submenu_and_refresh_route(app):
    from src.navigation import MENUS
    for main,items in MENUS.items():
        for route,title in items:
            navigate(app,main,title)
            assert_no_errors(app)
            assert app.query_params['page']==route
            assert not app.radio
            assert app.title[0].value==title
    app.run()
    assert app.title[0].value=='模型版本对比'
    app.query_params['page']='invalid'
    app.run();assert_no_errors(app)
    assert app.title[0].value=='装备态势'


def test_servo_pages_and_business_link(app):
    if not any(r.name.startswith('servo_') for r in list_runs()):pytest.skip('No real servo experiment yet')
    navigate(app,'健康分析','伺服执行机构诊断');assert_no_errors(app)
    slider=next(s for s in app.slider if s.label=='伺服已观测到的秒数')
    slider.set_value(5).run();assert_no_errors(app)
    assert next(m.value for m in app.metric if m.label=='RUL')=='无法评估'
    next(b for b in app.button if b.label=='建立伺服演示装备').click().run();assert_no_errors(app)
    assert app.title[0].value=='装备健康分析'
    servo_id=phm.platform.records('SELECT id FROM assets WHERE serial="DEMO-SERVO-001"')[0]['id']
    assert next(s for s in app.selectbox if s.label=='当前装备').value==servo_id
    # Existing selector state must not override a case-page deep link to another asset.
    other=phm.platform.records('SELECT id FROM assets WHERE kind="UAV"')[0]['id']
    next(s for s in app.selectbox if s.label=='当前装备').select(other).run()
    navigate(app,'健康分析','伺服执行机构诊断')
    next(b for b in app.button if b.label=='打开伺服装备健康分析').click().run();assert_no_errors(app)
    assert next(s for s in app.selectbox if s.label=='当前装备').value==servo_id
    navigate(app,'模型中心','伺服训练与验证');assert_no_errors(app)
    assert any(m.label=='测试记录 Macro-F1' for m in app.metric)
    navigate(app,'模型中心','模型目录与绑定')
    mid=next(r.name for r in list_runs() if r.name.startswith('servo_'))
    next(s for s in app.selectbox if s.label=='管理模型').select(mid).run();assert_no_errors(app)
    assert any(m.label=='记录 Macro-F1' for m in app.metric)
