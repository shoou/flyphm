"""Five business workspaces for the offline PHM prototype."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
from phm import platform as p
from phm.training import list_runs
from src import business_views as views
from src.navigation import sidebar_menu

st.set_page_config(page_title='PHM · 无人系统健康管理',page_icon='◈',layout='wide')
st.markdown('''<style>
.stApp {background:#f4f7fb;color:#152b43;}
.block-container {padding-top:4.5rem;padding-bottom:3rem;max-width:1500px;}
h1,h2,h3 {color:#162e49;letter-spacing:-.02em;}
[data-testid="stSidebar"] {background:#122b45;}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] p,[data-testid="stSidebar"] label {color:#e4edf6;}
.st-key-business_menu button {justify-content:flex-start;border-radius:8px;min-height:44px;}
.st-key-business_menu,.st-key-business_menu [data-testid="stVerticalBlock"] {gap:.3rem;}
.st-key-business_menu button > div {width:100%;justify-content:flex-start;}
.st-key-business_menu button > div > span {justify-content:flex-start;}
.st-key-business_menu button p {font-size:15px;font-weight:600;}
.st-key-business_menu button[kind="tertiary"] {color:#d2e1ef;}
.st-key-business_menu button[kind="tertiary"]:hover {background:#23425f;color:white;}
.st-key-business_menu button[kind="primary"] {background:#236a86;border-color:#236a86;}
.st-key-submenu {border-left:2px solid #356078;margin:0 0 12px 17px;padding-left:12px;width:calc(100% - 17px);box-sizing:border-box;}
.st-key-submenu button {min-height:38px;}
.st-key-submenu button p {font-size:14px;font-weight:400;}
.st-key-submenu button[kind="secondary"] {background:#e0f1f5;border-color:#e0f1f5;}
[data-testid="stSidebar"] .st-key-submenu button[kind="secondary"] p {color:#17425a;}
[data-testid="stMetric"] {background:white;border:1px solid #dce5ef;border-radius:12px;padding:18px;min-height:116px;}
[data-testid="stMetricLabel"] {color:#62758b;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:12px;}
.kicker {font-size:11px;font-weight:700;letter-spacing:.17em;color:#168b91;margin-bottom:8px;}
.context {color:#60738a;font-size:15px;margin-bottom:24px;}
[data-testid="stDataFrame"] {border-radius:10px;}
@media(max-width:760px){.block-container{padding:4.5rem 1rem 1rem;}[data-testid="stMetric"]{min-height:90px;padding:12px;}}
</style>''',unsafe_allow_html=True)
p.init();p.sync_models()
runs=list_runs();uav=[r for r in runs if r.name.startswith('uav_')];battery=[r for r in runs if r.name.startswith('battery_')]
with st.sidebar:
    st.markdown('## ◈ PHM PLATFORM')
    st.caption('无人系统 · 健康管理')
    route,page,view=sidebar_menu()
    st.divider()
    with st.expander('实验分析版本'):
        uav_name=st.selectbox('无人机实验',[r.name for r in uav]) if uav else None
        battery_name=st.selectbox('电池实验',[r.name for r in battery]) if battery else None
    st.caption('离线原型 / 公开真实数据\n\n')
    st.caption('伺服已支持公开台架诊断')
uav_run=p.ROOT/'runs'/uav_name if uav_name else None
battery_run=p.ROOT/'runs'/battery_name if battery_name else None
subtitles={'监测中心':'全局装备态势、异常证据与维修入口','装备管理':'装备档案、分系统结构与数据模型关联','健康分析':'从部件诊断到健康趋势与剩余寿命','维修管理':'从异常处置到复检关闭，保留完整履历','模型中心':'管理训练实验、验证指标和模型应用关系'}
st.markdown('<div class="kicker">UNMANNED SYSTEMS / HEALTH MANAGEMENT</div>',unsafe_allow_html=True)
st.caption(page+' / '+view)
st.title(view);st.markdown('<div class="context">'+subtitles[page]+'</div>',unsafe_allow_html=True)
if st.session_state.get('notice'):st.success(st.session_state.pop('notice'))
if page=='监测中心':views.monitor(view)
elif page=='装备管理':views.equipment(view)
elif page=='健康分析':views.health(uav_run,battery_run,view)
elif page=='维修管理':views.maintenance(uav_run,battery_run,view)
else:views.models(uav_run,battery_run,view)
