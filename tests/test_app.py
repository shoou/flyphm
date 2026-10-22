"""Exercise real-data pages when local experiment artifacts are available."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from phm.training import list_runs
import phm.events

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def app(tmp_path, monkeypatch):
    # UI verification never inserts test review events into the user's ledger.
    monkeypatch.setattr(phm.events, 'ROOT', tmp_path)
    return AppTest.from_file(str(ROOT/'src/app.py'), default_timeout=40).run()


def assert_no_errors(at):
    assert not at.exception, [e.message for e in at.exception]


def test_overview_and_training_page(app):
    assert_no_errors(app)
    app.radio[0].set_value('模型训练与验证').run()
    assert_no_errors(app)


def test_battery_censored_and_prefix_navigation(app):
    if not any(r.name.startswith('battery_') for r in list_runs()): pytest.skip('NASA experiment not present')
    app.radio[0].set_value('电池健康与寿命').run(); assert_no_errors(app)
    slider = next(s for s in app.slider if s.label=='已观测到的放电循环')
    slider.set_value(20).run(); assert_no_errors(app)
    next(c for c in app.checkbox if '未来' in c.label).check().run(); assert_no_errors(app)
    next(s for s in app.selectbox if s.label=='电池样本').select('B0007').run(); assert_no_errors(app)
    assert next(m.value for m in app.metric if m.label=='预测剩余循环')=='—'


def test_replay_seek_and_review(app):
    if not any(r.name.startswith('uav_') for r in list_runs()): pytest.skip('UAV experiment not present')
    app.radio[0].set_value('故障诊断回放').run(); assert_no_errors(app)
    app.slider[0].set_value(10).run(); assert_no_errors(app)
    next(b for b in app.button if b.label=='将当前窗口加入待复核事件').click().run(); assert_no_errors(app)
    next(b for b in app.button if b.label=='播放').click().run(); assert_no_errors(app)
    assert app.session_state['playing']
    next(b for b in app.button if b.label=='暂停').click().run(); assert_no_errors(app)
    assert not app.session_state['playing']
    app.radio[0].set_value('设备与案例总览').run(); assert_no_errors(app)
    next(s for s in app.selectbox if s.label=='处理状态').select('已确认')
    next(b for b in app.button if b.label=='保存处理记录').click().run(); assert_no_errors(app)
    assert phm.events.read_events().iloc[0].status=='已确认'
