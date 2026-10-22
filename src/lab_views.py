"""PHM research prototype — launch with streamlit run src/app.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from phm.data import ROOT, read_json, prepare_uav, prepare_battery
from phm.training import list_runs, train_uav, train_battery
from phm.events import read_events, save_event
from src.workflows import import_page, incidents_page, comparison_page

LABELS = {0:'正常',1:'1 个螺旋桨损伤',2:'2 个螺旋桨损伤',3:'3 个螺旋桨损伤',4:'4 个螺旋桨损伤'}

@st.cache_data
def table(path, modified):
    return pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path)

def load_table(path): return table(str(path), path.stat().st_mtime_ns)

def chart(fig, key=None):
    fig.update_layout(template='plotly_white', margin=dict(l=16,r=16,t=36,b=16), font=dict(family='Arial, sans-serif'),
                      paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h', y=1.12), height=340)
    st.plotly_chart(fig, width='stretch', key=key)

def metric_row(items):
    for col,(label,value,help_text) in zip(st.columns(len(items)), items):
        col.metric(label, value, help=help_text)

def empty_data():
    st.info('还没有完成的真实数据实验。先下载数据，再在“模型训练与验证”中准备数据并训练。')
    st.code('python scripts/download_data.py\npython -m phm.training --prepare', language='bash')


def render_lab(page, uav_run, battery_run):
    uav_name = uav_run.name if uav_run else None
    battery_name = battery_run.name if battery_run else None
    runs = list_runs()
    uav_runs = [r for r in runs if r.name.startswith('uav_')]
    battery_runs = [r for r in runs if r.name.startswith('battery_')]
    um = read_json(uav_run/'metrics.json') if uav_run else None
    bm = read_json(battery_run/'metrics.json') if battery_run else None
    if page == '设备与案例总览':
        st.markdown('<div class="hero"><h2>从运行信号，到可追溯的健康判断</h2><p>回放真实飞行案例 · 观察电池退化 · 核对独立测试结果</p></div>', unsafe_allow_html=True)
        metric_row([
            ('真实飞行案例', str(um['missions']) if um else '—', '公开室内飞行任务，不代表独立机体数量'),
            ('测试集 Macro-F1', f"{um['results']['test']['macro_f1']:.3f}" if um else '—', '仅测试任务，五分类窗口指标'),
            ('电池 RUL MAE', f"{bm['random_forest']['mae']:.1f} 循环" if bm else '—', '留一电池验证，失效前循环'),
            ('已完成实验', str(len(runs)), '每次运行保留配置、模型、划分和结果')])
        st.write('')
        left, right = st.columns([1.5,1])
        with left:
            st.subheader('案例覆盖')
            st.dataframe(pd.DataFrame([
                ['无人机动力 / 螺旋桨','真实室内飞行','分类与振动异常','已验证' if um else '待训练'],
                ['电池能源','真实实验室退化','SOH / 剩余放电循环','已验证' if bm else '待训练'],
                ['导航 / 通信 / 伺服','无对应数据','未知','待接入'],
                ['UGV 底盘 / 转向 / 制动','无实车故障数据','未知','待接入']],
                columns=['对象','来源','能力','状态']), hide_index=True, width='stretch')
        with right:
            st.subheader('当前评估边界')
            st.info('整机健康分：无法评估。当前飞行数据覆盖惯性信号，不能推断能源、通信、导航和执行器均正常。')
            st.caption('动力页面的 HI 是相对健康训练样本的振动偏离示意指标，尚未标定到实际失效概率。')
        if um:
            quality = read_json(uav_run/'data_quality.json')
            with st.expander('数据质量与来源'):
                st.write(quality['source']); st.caption(quality['label_scope'])
                st.dataframe(pd.DataFrame(quality['missions']), hide_index=True, width='stretch')
        if not runs: empty_data()

    elif page == '历史窗口复核':
        st.subheader('告警复核与处理记录')
        events = read_events()
        if events.empty: st.caption('在故障诊断回放中记录待复核事件；处理意见会保存在本地。')
        else:
            st.dataframe(events, hide_index=True, width='stretch')
            event_id = st.selectbox('更新事件', events.id.tolist())
            ev = events[events.id == event_id].iloc[0]
            with st.form('event_review'):
                status = st.selectbox('处理状态', ['待复核','已确认','误报','已处理'], index=['待复核','已确认','误报','已处理'].index(ev.status))
                note = st.text_area('复核 / 维修意见', value=ev.note)
                if st.form_submit_button('保存处理记录'):
                    save_event(ev.run_id, ev.mission_id, ev.timestamp_s, ev.predicted_label, status, note)
                    st.success('已保存'); st.rerun()
            st.download_button('导出处理记录 CSV', events.to_csv(index=False).encode('utf-8-sig'), 'events.csv', 'text/csv')

    elif page == '故障诊断回放':
        if not uav_run: empty_data(); st.stop()
        w = load_table(uav_run/'predictions.parquet')
        role = st.selectbox('案例集合', ['test','validation','train'], format_func=lambda x:{'test':'独立测试任务','validation':'验证任务','train':'训练任务'}[x])
        subset = w[w.split == role]
        mission = st.selectbox('飞行任务', sorted(subset.mission_id.unique()))
        g = subset[subset.mission_id == mission].sort_values('timestamp_s').reset_index(drop=True)
        st.caption(f"真实飞行 · {role} · 任务级标签：{LABELS[int(g.label.iloc[0])]} · {uav_run.name}")
        st.info('损伤标签适用于整个任务，不包含飞行中故障起始时刻，因此不报告检测延迟。模型概率尚未校准。')
        if um.get('evaluation_status') == 'previously_seen_test_regression':
            st.caption('此处旧测试任务仅用于回归回放；冻结参数后的新增任务结果请见“模型版本对比”。')
        token = f'{uav_name}/{mission}'
        if st.session_state.get('replay_token') != token:
            st.session_state.update(replay_token=token, cursor=0, playing=False)
        def toggle_playback():
            st.session_state.playing = not st.session_state.playing
        def reset_playback():
            st.session_state.cursor = 0
            st.session_state.playing = False
        c1,c2,c3 = st.columns([1,1,3])
        c1.button('暂停' if st.session_state.playing else '播放', width='stretch', key='play_pause', on_click=toggle_playback)
        c2.button('回到起点', width='stretch', on_click=reset_playback)
        speed = c3.select_slider('回放倍速', options=[1,2,5,10], value=1)
        trace_path = uav_run/'telemetry.parquet'
        trace = load_table(trace_path) if trace_path.exists() else pd.DataFrame()
        if not trace.empty: trace = trace[trace.mission_id == mission]
        @st.fragment(run_every=1. if st.session_state.playing else None)
        def playback():
            if st.session_state.playing:
                st.session_state.cursor = min(len(g)-1, st.session_state.cursor+speed)
            idx = st.slider('回放窗口', 0, len(g)-1, key='cursor')
            row = g.iloc[idx]; shown = g.iloc[:idx+1]
            metric_row([('时间', f'{row.timestamp_s:.0f} s','窗口结束时间'),
                ('模型判断',LABELS[int(row.predicted_label)],'故障类别由模型推理'),
                ('类别概率',f'{row.class_probability:.1%}','随机森林未校准概率'),
                ('动力 HI（示意）',f'{row.propulsion_hi_proxy:.0f} / 100','仅反映振动偏离；非整机健康分')])
            action = ('分类模型未检出损伤，但异常检测已触发；建议复核振动和工况。'
                      if row.predicted_label == 0 and row.get('confirmed_alert',row.anomaly_flag) else row.recommended_action)
            st.write(action)
            if 'confirmed_alert' in row:
                st.caption(f"代理工况：{row.get('operating_regime','unknown')} · 原始越限：{'是' if row.raw_anomaly_flag else '否'} · 连续确认告警：{'是' if row.confirmed_alert else '否'}")
            st.caption('导航、能源、通信、伺服：未知。任务风险：尚未建立经过验证的风险模型。')
            left,right = st.columns(2)
            with left:
                if not trace.empty:
                    visible = trace[trace.timestamp_s <= row.timestamp_s]
                    chart(px.line(visible,x='timestamp_s',y=['acc_x','acc_y','acc_z'], labels={'timestamp_s':'时间 / s','value':'加速度 / m/s²'},title='惯性遥测 · 历史回放'), 'telemetry')
                else: st.info('当前实验无原始遥测曲线。')
            with right:
                fig = px.line(shown, x='timestamp_s', y='anomaly_score', title='异常分数与验证集阈值', labels={'timestamp_s':'时间 / s','anomaly_score':'异常分数'})
                if 'anomaly_threshold' in shown:
                    fig.add_scatter(x=shown.timestamp_s,y=shown.anomaly_threshold,name='工况阈值',line=dict(dash='dash',color='#db704d'))
                else:
                    fig.add_hline(y=um['thresholds']['anomaly'], line_dash='dash', line_color='#db704d')
                chart(fig,'anomaly')
            fig = go.Figure()
            fig.add_scatter(x=shown.timestamp_s,y=shown.predicted_label,mode='lines',name='模型判断',line_shape='hv')
            fig.add_scatter(x=shown.timestamp_s,y=shown.label,mode='lines',name='任务真实标签',line=dict(dash='dash'))
            fig.update_layout(title='故障类别时间线', xaxis_title='时间 / s',yaxis=dict(tickvals=list(LABELS),ticktext=list(LABELS.values())))
            chart(fig,'timeline')
            with st.expander('当前窗口的判断依据', expanded=True):
                st.dataframe(pd.DataFrame({'指标':['加速度模长标准差','健康验证集规则阈值','角速度模长 RMS','异常阈值超出'],
                    '值':[f'{row.acc_norm_std:.4f} m/s²',f"{um['thresholds']['acc_norm_std']:.4f} m/s²",f'{row.gyro_norm_rms:.4f} rad/s', '是' if row.get('raw_anomaly_flag',row.anomaly_flag) else '否']}),hide_index=True,width='stretch')
                st.caption('这些是传感器证据及阈值比较，不能单独证明具体损伤位置。')
            if st.button('将当前窗口加入待复核事件'):
                save_event(uav_run.name,mission,row.timestamp_s,row.predicted_label,'待复核',action)
                st.success('已记录，可在总览页面补充复核及维修意见。')
            if idx == len(g)-1 and st.session_state.playing:
                st.session_state.playing = False; st.rerun()
        playback()
        st.download_button('导出本任务预测 CSV',g.to_csv(index=False).encode('utf-8-sig'),f'{mission}.csv','text/csv')

    elif page == '电池健康与寿命':
        curve_path = battery_run/'curves.parquet' if battery_run else ROOT/'data/processed/battery/curves.parquet'
        if not curve_path.exists(): empty_data(); st.stop()
        curves = load_table(curve_path)
        bid = st.selectbox('电池样本', sorted(curves.battery_id.unique()))
        g = curves[curves.battery_id == bid].sort_values('cycle_idx')
        cutoff = st.slider('已观测到的放电循环', 1, int(g.cycle_idx.max()), min(60,int(g.cycle_idx.max())))
        observed = g[g.cycle_idx <= cutoff]
        show_future = st.checkbox('显示未来真实容量（仅用于验证）', value=False)
        pred = load_table(battery_run/'predictions.parquet') if battery_run else pd.DataFrame()
        current = pred[(pred.battery_id == bid)&(pred.cycle_idx == cutoff)] if not pred.empty else pd.DataFrame()
        row = current.iloc[0] if len(current) else None
        metric_row([('观测 SOH',f'{observed.soh.iloc[-1]:.1%}','实测容量 / 额定容量 2 Ah；不是模型估计精度'),
            ('当前容量',f'{observed.capacity_ah.iloc[-1]:.3f} Ah','截至当前循环的真实容量'),
            ('预测剩余循环',f'{row.predicted_rul:.1f}' if row is not None else '—','该电池未参与本折模型训练'),
            ('探索性预测区间',f'{row.lower:.0f} – {row.upper:.0f}' if row is not None else '—','基于训练电池留一残差，小样本下不保证覆盖率')])
        fig = go.Figure()
        fig.add_scatter(x=observed.cycle_idx,y=observed.capacity_ah,name='已观测容量',line=dict(color='#078681',width=3))
        if show_future: fig.add_scatter(x=g[g.cycle_idx>cutoff].cycle_idx,y=g[g.cycle_idx>cutoff].capacity_ah,name='未来真实容量（验证）',line=dict(color='#94a3b8',dash='dot'))
        fig.add_hline(y=1.4,line_color='#db704d',line_dash='dash',annotation_text='失效判据 1.4 Ah')
        fig.add_vline(x=cutoff,line_dash='dot',line_color='#94a3b8')
        if row is not None:
            fig.add_vrect(x0=cutoff+row.lower,x1=cutoff+row.upper,fillcolor='#13a89e',opacity=.12,line_width=0)
            fig.add_scatter(x=[cutoff+row.predicted_rul],y=[1.4],mode='markers',name='预测阈值到达循环',marker=dict(size=12,color='#db704d'))
        fig.update_layout(title='容量退化与预测失效窗口',xaxis_title='放电循环',yaxis_title='容量 / Ah')
        chart(fig)
        if row is None:
            st.info('当前循环没有可验证的 RUL 输出：可能历史不足 10 个循环、该电池未观察到失效，或已到达失效阈值。保留容量和 SOH 展示。')
        else:
            st.caption('阴影是失效循环的探索性预测区间；不表示未来容量曲线的置信带。每个预测只使用截止当前的循环特征。')
            st.write('维护建议：依据剩余放电循环与任务需求安排容量复测；本结果不能换算为无人机剩余飞行小时。')
        if not pred.empty:
            history = pred[(pred.battery_id == bid)&(pred.cycle_idx <= cutoff)]
            if not history.empty:
                fig = go.Figure()
                fig.add_scatter(x=history.cycle_idx,y=history.upper,line=dict(width=0),showlegend=False)
                fig.add_scatter(x=history.cycle_idx,y=history.lower,fill='tonexty',fillcolor='rgba(19,168,158,.15)',line=dict(width=0),name='探索性区间')
                fig.add_scatter(x=history.cycle_idx,y=history.predicted_rul,name='随机森林 RUL')
                fig.add_scatter(x=history.cycle_idx,y=history.linear_rul,name='线性外推基线',line=dict(dash='dot'))
                if show_future: fig.add_scatter(x=history.cycle_idx,y=history.true_rul,name='真实 RUL（事后验证）',line=dict(color='#273e59',dash='dash'))
                fig.update_layout(title='历次预测结果',xaxis_title='观测截止循环',yaxis_title='剩余放电循环')
                chart(fig)
        st.download_button('导出容量曲线 CSV',g.to_csv(index=False).encode('utf-8-sig'),f'{bid}_capacity.csv','text/csv')

    elif page == '数据导入与推理':
        import_page(uav_run)
    elif page == '异常事件与复核':
        incidents_page(uav_run)
    elif page == '模型版本对比':
        comparison_page(uav_runs)
    else:
        st.caption('数据来源、分组划分、模型参数与评估结果随每次实验一起保存。训练和下载分开执行，演示时可直接加载已完成实验。')
        with st.expander('运行新实验', expanded=not runs):
            task = st.selectbox('训练任务',['uav','battery'],format_func=lambda x:'无人机螺旋桨分类' if x=='uav' else '电池剩余循环预测')
            trees = st.select_slider('随机森林树数量', options=[80,120,160,240],value=160)
            if st.button('准备本地原始数据并训练',type='primary'):
                try:
                    with st.status('正在检查数据并训练…',expanded=True) as status:
                        st.write('核对数据来源与质量')
                        (prepare_uav if task=='uav' else prepare_battery)()
                        st.write('运行分组验证；测试数据不参与阈值选择')
                        result = (train_uav if task=='uav' else train_battery)(trees=trees)
                        status.update(label='训练完成',state='complete')
                    st.success(f'实验已保存：{result.name}。刷新实验列表后可选择。')
                except Exception as e: st.error(f'训练未完成：{e}')
            if st.button('刷新实验列表'): st.rerun()
            st.code('python scripts/download_data.py\npython -m phm.training --prepare',language='bash')
        tabs = st.tabs(['无人机诊断','电池寿命','实验与数据说明'])
        with tabs[0]:
            if um:
                scores = pd.DataFrame([{ '集合':role, 'Macro-F1':um['results'][role]['macro_f1'], '准确率':um['results'][role]['accuracy']} for role in ['train','validation','test']])
                chart(px.bar(scores,x='集合',y=['Macro-F1','准确率'],barmode='group',range_y=[0,1],title='训练 / 验证 / 独立测试'))
                left,right = st.columns(2)
                with left:
                    result = um['results']['test']
                    chart(px.imshow(result['confusion_matrix'],x=[LABELS[i] for i in result['labels']],y=[LABELS[i] for i in result['labels']],text_auto=True,color_continuous_scale='Teal',labels={'x':'预测','y':'真实'},title='独立测试混淆矩阵'))
                with right:
                    imp = load_table(uav_run/'feature_importance.csv').head(10).sort_values('importance')
                    chart(px.bar(imp,x='importance',y='feature',orientation='h',title='全局特征重要性（非因果解释）'))
                if 'alert_improvement' in um:
                    st.info('下方异常指标保留原始基线。连续确认后的效果及新增任务评价，请查看“模型版本对比”。')
                ev = um['results']['anomaly_events']
                metric_row([('测试任务',str(um['split_counts']['test']),'完整任务隔离'),
                    ('误报事件 / 小时',f"{ev['false_alarm_events_per_hour']:.1f}" if ev['false_alarm_events_per_hour'] is not None else '—','连续异常窗口合并；有效健康窗口时长作为分母'),
                    ('故障任务检出率',f"{ev['fault_mission_recall']:.1%}" if ev['fault_mission_recall'] is not None else '—','任务内至少一个异常窗口，不代表及时检出'),
                    ('训练验证耗时',f"{um['duration_s']:.1f} s",'不含数据下载与特征处理')])
                st.caption(f"健康测试曝光 {ev['healthy_exposure_hours']:.3f} 小时；检测延迟不可用。短数据集的每小时误报率不稳定。")
                st.dataframe(pd.DataFrame({'模型':['加速度波动阈值（二分类）','随机森林（二分类）'], '测试 Macro-F1':[um['results']['binary_rule']['macro_f1'],um['results']['binary_rf']['macro_f1']]}),hide_index=True,width='stretch')
                report = um['results']['test']['report']
                st.dataframe(pd.DataFrame({LABELS[int(k)]:v for k,v in report.items() if k.isdigit()}).T,width='stretch')
                st.dataframe(load_table(uav_run/'per_mission.csv'),hide_index=True,width='stretch')
                st.download_button('导出无人机测试结果 CSV',load_table(uav_run/'predictions.parquet').query("split == 'test'").to_csv(index=False).encode('utf-8-sig'),'uav_test.csv','text/csv')
            else: empty_data()
        with tabs[1]:
            if bm:
                interval = bm['interval']
                metric_row([('RUL MAE',f"{bm['random_forest']['mae']:.2f} 循环",'按窗口汇总'),
                    ('RUL RMSE',f"{bm['random_forest']['rmse']:.2f} 循环",'按窗口汇总'),
                    ('区间实测覆盖率',f"{interval['empirical_coverage']:.1%}",'留出电池上的实际覆盖率，非保证'),
                    ('平均区间宽度',f"{interval['mean_width']:.1f} 循环",'区间越宽通常覆盖越高')])
                st.warning('预测区间仅为小样本探索结果。SOH 来自容量观测，不是模型预测成绩。')
                st.dataframe(pd.DataFrame([{'模型':'随机森林',**bm['common_support']['random_forest']},{'模型':'线性外推',**bm['common_support']['linear']}]),hide_index=True,width='stretch')
                st.caption('上表在两种模型均有输出的相同循环上比较；线性斜率非负时不输出寿命。')
                per = pd.DataFrame(bm['per_battery']).T.reset_index(names='battery_id')
                chart(px.bar(per,x='battery_id',y=['mae','rmse'],barmode='group',title='逐电池留一验证误差',labels={'value':'放电循环'}))
                st.dataframe(pd.DataFrame(bm['per_stage']).T,width='stretch')
                st.dataframe(pd.DataFrame(bm['states']),hide_index=True,width='stretch')
                st.download_button('导出电池留一预测 CSV',load_table(battery_run/'predictions.parquet').to_csv(index=False).encode('utf-8-sig'),'battery_predictions.csv','text/csv')
            else: empty_data()
        with tabs[2]:
            report_path = ROOT/'runs/validation_report.html'
            if report_path.exists():
                st.download_button('下载最近汇总的离线验证报告 HTML', report_path.read_bytes(), 'PHM_validation_report.html', 'text/html')
                st.caption('汇总报告保留生成时的实验编号；重新训练后运行 scripts/summarize_results.py 更新。')
            for run,metrics in [(uav_run,um),(battery_run,bm)]:
                if run:
                    st.subheader(run.name)
                    st.json(read_json(run/'config.json'),expanded=False)
                    st.json(read_json(run/'data_quality.json'),expanded=False)
                    if (run/'split.json').exists(): st.json(read_json(run/'split.json'),expanded=False)
                    for limitation in metrics['limitations']: st.caption(limitation)
                    st.download_button('下载实验报告 JSON',json.dumps(metrics,ensure_ascii=False,indent=2),f'{run.name}.json','application/json',key=run.name)
            st.caption('数据入口：UAV Realistic Fault Dataset（TII）和 NASA PCoE Battery。原始文件保留本地，不随代码分发。')
            st.link_button('无人机数据来源','https://github.com/tiiuae/UAV-Realistic-Fault-Dataset')
            st.link_button('NASA 数据来源','https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/')
