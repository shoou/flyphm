"""Servo case review and independent validation pages."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from phm import platform as p
from phm import servo
from phm.training import list_runs
from src.lab_views import chart,metric_row,load_table
from src.navigation import go


def choose_run():
    runs=[r for r in list_runs() if r.name.startswith('servo_')]
    if not runs:
        st.info('尚无完成的伺服实验，请先下载公开数据，再准备数据并训练。')
        st.code('.venv/bin/python scripts/download_servo.py\n.venv/bin/python -m phm.servo --prepare',language='bash')
        return None
    return p.ROOT/'runs'/st.selectbox('伺服模型版本',[r.name for r in runs],key='servo_run')


def case_page():
    run=choose_run()
    if run is None:return
    st.caption('公开无负载台架 · 振动仪 WAV · 四种运行状态 · 跨转速测试，不代表跨设备验证')
    st.caption('数据：Zhilevski、Slavov、Yordanov（2026），[Mendeley V13](https://data.mendeley.com/datasets/g28trvywnx/13)，CC BY 4.0。')
    w=load_table(run/'predictions.parquet')
    c1,c2=st.columns(2)
    role=c1.selectbox('伺服案例集合',['test','validation','train'],format_func=lambda x:{'test':'独立测试工况','validation':'模型选择验证工况','train':'训练工况'}[x])
    labels=sorted(w[w.split==role].label.unique())
    state=c2.selectbox('案例真实状态',labels,format_func=lambda x:servo.LABELS[int(x)])
    records=w[(w.split==role)&(w.label==state)].record_id.unique()
    rid=st.selectbox('伺服记录',sorted(records))
    g=w[w.record_id==rid].sort_values('timestamp_s');end=int(g.timestamp_s.max())
    cutoff=st.slider('伺服已观测到的秒数',int(g.timestamp_s.min()),end,end) if int(g.timestamp_s.min())<end else end
    observed=g[g.timestamp_s<=cutoff];last=observed.iloc[-1];events=servo.fault_events(observed)
    quality=p.read_json(run/'data_quality.json');q=next(r for r in quality['records'] if r['record_id']==rid)
    gaps=[x for x in q['unknown_windows'] if x['second']<=cutoff]
    if gaps:st.warning(f'观测范围内有 {len(gaps)} 个窗口质量不足，未输出分类；下方为最新有效窗口结果。')
    timeline=observed.set_index('timestamp_s').reindex(range(2,cutoff+1)).rename_axis('timestamp_s').reset_index()
    items=[('当前诊断',servo.LABELS[int(last.predicted_label)],'模型仅使用振动特征'),('分类概率',f'{last.class_probability:.1%}','未经概率校准'),('复核事件',str(len(events)),'连续 3 个相同异常分类窗口'),('RUL','无法评估','没有寿命退化标签')]
    for col,(label,value,help_text) in zip(st.columns([1.5,1,1,1.1]),items):col.metric(label,value,help=help_text)
    st.caption(f"指令转速 {int(last.speed_pct)}% · {'换向工况' if last.reversal else '不换向工况'} · 真实状态为整段记录标签，不包含故障起点。")
    if int(last.predicted_label)!=int(state):st.warning('当前有效窗口与公开案例标注不一致，是已知误判，须人工复核。')
    if events:st.warning('已触发人工复核建议：'+ '、'.join(sorted(set(e['evidence']['condition'] for e in events))))
    else:st.info('当前观测未触发连续异常复核规则；不等于整机健康或无故障保证。')
    left,right=st.columns(2)
    with left:
        st.subheader('振动特征趋势')
        feature=st.selectbox('伺服趋势指标',['rms','crest','kurtosis','dominant_hz','class_probability'])
        chart(px.line(timeline,x='timestamp_s',y=feature,labels={'timestamp_s':'观测时间（秒）'}))
    with right:
        st.subheader('模型诊断时间线')
        fig=px.line(timeline,x='timestamp_s',y='predicted_label',line_shape='hv',labels={'timestamp_s':'观测时间（秒）','predicted_label':'模型类别'})
        fig.update_yaxes(tickvals=list(servo.LABELS),ticktext=list(servo.LABELS.values()),range=[-.2,3.2]);chart(fig)
    with st.expander('原始振动包络与质量检查'):
        signal=load_table(run/'waveforms.parquet');signal=signal[(signal.record_id==rid)&(signal.timestamp_s+.01<=cutoff+1e-9)]
        chart(px.line(signal,x='timestamp_s',y=['minimum','maximum'],labels={'timestamp_s':'时间（秒）','value':'数字归一化幅值'}))
        quality=p.read_json(run/'data_quality.json');q=next(r for r in quality['records'] if r['record_id']==rid)
        st.json(q);st.caption('包络为 10 ms 内极值，非完整高频波形；幅值未标定为物理加速度。')
    with st.expander('异常证据与建议'):
        st.json(events)
        st.write('建议人工检查固定螺栓、调节参数和振动；不自动调整驱动参数。数据无同步指令/反馈轨迹，不展示虚构的跟踪误差。')
    st.download_button('导出伺服观测窗口',observed.to_csv(index=False).encode('utf-8-sig'),file_name=rid+'_observed.csv',mime='text/csv')
    st.subheader('进入装备与维修流程')
    existing=p.records('SELECT id FROM assets WHERE serial="DEMO-SERVO-001" AND is_demo=1')
    if not existing:
        if st.button('建立伺服演示装备',type='primary'):
            try:aid=p.seed_servo_demo();st.session_state['active_asset']=aid;st.session_state['health_asset']=aid;go('health');st.rerun()
            except ValueError as exc:st.error(str(exc))
    else:
        st.caption('案例页仅回放实验；装备分析使用装备管理中保存的绑定与截止位置。')
        if st.button('打开伺服装备健康分析',type='primary'):
            st.session_state['active_asset']=existing[0]['id'];st.session_state['health_asset']=existing[0]['id'];go('health');st.rerun()


def training_page():
    with st.expander('准备数据与重新训练'):
        st.caption('固定 V13 振动波形。先按预设工况划分，再仅用验证集选择随机森林或逻辑回归；测试集不参与选择。')
        available=(servo.RAW/'manifest.json').exists()
        if not available:st.code('.venv/bin/python scripts/download_servo.py',language='bash')
        if st.button('准备伺服数据并训练',disabled=not available):
            try:
                with st.spinner('检查原始哈希、提取特征并完成跨转速验证…'):
                    run=servo.train(servo.prepare());p.sync_models()
                st.session_state['servo_run']=run.name;st.success('实验已完成，保留已有版本。')
            except Exception as exc:st.error('伺服训练未完成：'+str(exc))
    run=choose_run()
    if run is None:return
    m=p.read_json(run/'metrics.json');test=m['results']['test']
    metric_row([('真实记录',str(m['recordings']),'来自单台架，不能视为独立电机数量'),('测试记录 Macro-F1',f"{test['records']['macro_f1']:.3f}",'完整测试录音多数投票'),('测试窗口 Macro-F1',f"{test['windows']['macro_f1']:.3f}",'1 秒非重叠窗口'),('测试记录',str(test['n_records']),'按转速整体留出')])
    st.info('测试转速：20%、50%；验证转速：10%、40%；其余转速用于训练。同一工况的全部窗口始终在同一集合。')
    st.caption('使用转速工况：'+ '、'.join(str(x)+'%' for x in m.get('selected_speeds',[])))
    if test['records']['macro_f1']<.8:st.warning('当前基线仍有明显误判，适合展示研究流程，尚不足以作为现场自动诊断依据。')
    st.caption('所选模型：'+m['selected_model']+'；speed、标签、文件名与工况标识不作为模型输入。')
    st.subheader('训练 / 验证 / 测试效果')
    rows=[]
    for role,title in [('train','训练'),('validation','验证'),('test','测试')]:
        r=m['results'][role];rows.append({'集合':title,'记录数':r['n_records'],'窗口数':r['n_windows'],'记录 Macro-F1':r['records']['macro_f1'],'窗口 Macro-F1':r['windows']['macro_f1'],'窗口准确率':r['windows']['accuracy'],'多数类基线 F1':r['baseline_windows']['macro_f1']})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    unit=st.selectbox('测试指标粒度',['records','windows'],format_func=lambda x:'整段记录' if x=='records' else '时间窗口')
    left,right=st.columns(2)
    with left:
        st.subheader('独立测试混淆矩阵')
        chart(px.imshow(test[unit]['confusion_matrix'],x=list(servo.LABELS.values()),y=list(servo.LABELS.values()),text_auto=True,color_continuous_scale='Blues',labels={'x':'预测状态','y':'真实状态','color':'数量'}))
    with right:
        st.subheader('各状态召回与精确率')
        report=test[unit]['report'];rows=[{'状态':name,'精确率':report[str(k)]['precision'],'召回率':report[str(k)]['recall'],'F1':report[str(k)]['f1-score'],'数量':report[str(k)]['support']} for k,name in servo.LABELS.items()]
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.subheader('逐记录结果与误判')
    rp=pd.read_csv(run/'record_predictions.csv');te=rp[rp.split=='test'].copy()
    only_errors=st.checkbox('仅显示误判记录',value=True)
    shown=te[te.label!=te.predicted_label] if only_errors else te
    if shown.empty:st.caption('当前筛选没有误判记录；样本量有限，不代表现场零误报。')
    else:
        shown=shown.copy();shown['label']=shown.label.map(servo.LABELS);shown['predicted_label']=shown.predicted_label.map(servo.LABELS)
        st.dataframe(shown.rename(columns={'record_id':'记录','label':'真实状态','predicted_label':'预测状态','speed_pct':'转速百分比','vote_fraction':'多数票占比'}),hide_index=True,width='stretch')
    with st.expander('模型选择、特征贡献与数据质量'):
        st.dataframe(pd.DataFrame(m['selection']),hide_index=True)
        importance=pd.read_csv(run/'feature_importance.csv').head(12)
        chart(px.bar(importance,x='importance',y='feature',orientation='h'))
        st.json(m['per_test_speed']);st.json(p.read_json(run/'data_quality.json'))
    for limitation in m['limitations']:st.caption('• '+limitation)
    st.markdown('[数据来源与引用：Mendeley V13 · CC BY 4.0](https://data.mendeley.com/datasets/g28trvywnx/13)')
    st.download_button('导出伺服验证指标',p.encode(m),file_name=run.name+'_metrics.json',mime='application/json')
    st.download_button('导出逐记录验证结果',rp.to_csv(index=False).encode('utf-8-sig'),file_name=run.name+'_records.csv',mime='text/csv')
    if (run/'validation_report.html').exists():st.download_button('下载伺服验证报告', (run/'validation_report.html').read_bytes(),file_name='servo_validation_report.html',mime='text/html')
