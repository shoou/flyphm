"""CSV, event lifecycle and experiment-comparison views."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from phm.data import ROOT, read_json, read_tii, CHANNELS
from phm.imports import read_csv_bytes, inspect_csv, save_import, infer_import, REQUIRED, OPTIONAL, UNITS
from phm.events import import_incidents, read_incidents, update_incident, incident_history
from phm.experiments import improve_alerts, compare_runs, comparison_html


def import_page(uav_run):
    st.caption('当前支持六轴 IMU CSV：加速度、角速度、时间、设备和任务标识。至少约 50 Hz，模型使用 2 秒窗口；低频遥测或其他设备可先查看质量报告。')
    st.warning('用户数据只做适用范围内的推理，不自动成为真实验证成绩。没有标签也能推理；缺失信号和断流窗口显示无法评估。')
    upload=st.file_uploader('上传 CSV（UTF-8，最大 50 MB）',type=['csv'])
    if st.button('载入现有真实飞行任务作为导入示例'):
        manifest=read_json(ROOT/'data/raw/tii/manifest.json'); entry=manifest['files'][0]
        frame,_=read_tii(ROOT/'data/raw/tii'/entry['path'])
        frame['device_id']='TII_example';frame['mission_id']=Path(entry['path']).parent.name
        st.session_state['csv_example']=frame.to_csv(index=False).encode()
    payload=upload.getvalue() if upload else st.session_state.get('csv_example')
    st.download_button('下载空白字段模板',','.join(REQUIRED)+'\n','imu_template.csv','text/csv')
    if payload:
        if upload is None: st.caption('示例来自已使用过的公开任务，仅演示导入流程，不是新增独立测试。')
        try:
            raw=read_csv_bytes(payload)
            st.dataframe(raw.head(12),hide_index=True,width='stretch')
            token=hashlib.sha256(payload).hexdigest()[:12]
            mapping={}; columns=['（未提供）',*raw.columns.tolist()]
            with st.expander('字段映射与物理单位',expanded=True):
                cols=st.columns(3)
                for i,field in enumerate(REQUIRED+OPTIONAL):
                    value=cols[i%3].selectbox(field,columns,index=columns.index(field) if field in columns else 0,key=f'map_{token}_{field}')
                    mapping[field]=None if value=='（未提供）' else value
                u1,u2,u3=st.columns(3)
                units={'time':u1.selectbox('原时间单位',list(UNITS['time']),key=f'time_{token}'),
                       'acc':u2.selectbox('原加速度单位',list(UNITS['acc']),key=f'acc_{token}'),
                       'gyro':u3.selectbox('原角速度单位',list(UNITS['gyro']),key=f'gyro_{token}')}
            normalized,quality=inspect_csv(payload,mapping,units)
            st.subheader('导入前质量检查')
            for error in quality['blocking_errors']: st.error(error)
            for warning in quality['warnings']: st.caption(warning)
            if quality['missions']: st.dataframe(pd.DataFrame(quality['missions']),hide_index=True,width='stretch')
            with st.expander('完整质量报告'): st.json(quality)
            if st.button('保存此数据集',disabled=bool(quality['blocking_errors']),type='primary'):
                folder=save_import(payload,mapping,units,upload.name if upload else 'existing_real_mission.csv')
                st.session_state['latest_import']=folder.name;st.success(f'已保存：{folder.name}')
        except (ValueError,KeyError,pd.errors.ParserError) as exc: st.error(str(exc))
    folders=sorted((ROOT/'data/imports').glob('*/manifest.json'),reverse=True)
    if not folders: return
    options=[p.parent.name for p in folders]
    selected=st.selectbox('已保存的数据集',options,index=options.index(st.session_state['latest_import']) if st.session_state.get('latest_import') in options else 0)
    folder=ROOT/'data/imports'/selected
    st.json(read_json(folder/'manifest.json'),expanded=False)
    if uav_run is None: st.info('请先完成无人机模型训练');return
    st.caption(f'推理模型：{uav_run.name}')
    if st.button('运行无标签推理'):
        try:
            with st.spinner('生成有效窗口并推理…'): infer_import(folder,uav_run)
        except Exception as exc: st.error(f'推理失败：{exc}')
    output=folder/('inference_'+uav_run.name)
    if (output/'status.json').exists():
        status=read_json(output/'status.json')
        st.dataframe(pd.DataFrame(status['missions']),hide_index=True,width='stretch')
        if status['status']=='unknown': st.warning(status['reason']);return
        predictions=pd.read_parquet(output/'predictions.parquet')
        mission=st.selectbox('导入任务',predictions.mission_id.unique())
        shown=predictions[predictions.mission_id==mission]
        if (output/'window_quality.parquet').exists():
            grid=pd.read_parquet(output/'window_quality.parquet')
            shown=grid[grid.mission_id==mission].merge(shown,on=['mission_id','timestamp_s'],how='left')
        st.plotly_chart(px.line(shown,x='timestamp_s',y=['anomaly_score','anomaly_threshold'],title='导入数据 · 异常分数与工况阈值'),width='stretch')
        columns=['timestamp_s','predicted_label','class_probability','confirmed_alert']+(['window_status'] if 'window_status' in shown else [])
        st.dataframe(shown[columns],hide_index=True,width='stretch')
        st.caption('unknown 窗口未执行推理；曲线在缺失处断开，不代表正常。')
        events=pd.read_parquet(output/'alarm_events.parquet')
        st.dataframe(events,hide_index=True,width='stretch')
        st.download_button('导出导入数据预测',predictions.to_csv(index=False).encode('utf-8-sig'),'import_predictions.csv','text/csv')
        st.download_button('导出导入质量报告',json.dumps(read_json(folder/'quality.json'),ensure_ascii=False,indent=2),'import_quality.json','application/json')
        if not events.empty and st.button('将导入异常事件加入复核台账'):
            import_incidents(f'{selected}@{uav_run.name}',events);st.success('已加入，数据来源标识随事件保存。')


def incidents_page(uav_run):
    st.caption('连续异常达到确认条件后才建立事件；短暂恢复可合并为同一事件。复核、维修、复检的每次变更均保留历史。')
    if uav_run and (uav_run/'alarm_events.parquet').exists():
        events=pd.read_parquet(uav_run/'alarm_events.parquet')
        predictions=pd.read_parquet(uav_run/'predictions.parquet')
        role=st.selectbox('事件集合',['test','validation','train'])
        events=events[events.mission_id.isin(predictions[predictions.split==role].mission_id)]
        st.dataframe(events,hide_index=True,width='stretch')
        if not events.empty:
            chosen=st.selectbox('回放异常事件',events.event_id.tolist())
            e=events[events.event_id==chosen].iloc[0]
            context=predictions[(predictions.mission_id==e.mission_id)&(predictions.timestamp_s>=e.start_s-10)&(predictions.timestamp_s<=e.end_s+10)]
            fig=px.line(context,x='timestamp_s',y=['anomaly_score','anomaly_threshold'],title='事件前后证据')
            fig.add_vline(x=e.confirmed_s,line_dash='dash',annotation_text='告警确认')
            fig.add_vrect(x0=e.start_s,x1=e.end_s,fillcolor='#d88855',opacity=.15,line_width=0)
            st.plotly_chart(fig,width='stretch')
            st.caption(f'首次候选异常 {e.start_s:.1f}s；确认 {e.confirmed_s:.1f}s；结束 {e.end_s:.1f}s。确认耗时不等于真实故障检测延迟。结束原因：{e.end_reason}')
            if st.button('将所选完整事件加入复核台账'):
                import_incidents(uav_run.name,events[events.event_id==chosen]);st.success('已加入；重复加入不会覆盖已有处理意见。')
            st.download_button('导出事件清单',events.to_csv(index=False).encode('utf-8-sig'),'alarm_events.csv','text/csv')
    else: st.info('先在“模型版本对比”中生成带连续确认逻辑的告警版本。')
    st.subheader('事件处理台账')
    ledger=read_incidents()
    if ledger.empty: st.caption('暂无待复核完整事件。原有窗口级记录仍保留在总览页面。');return
    st.dataframe(ledger,hide_index=True,width='stretch')
    key=st.selectbox('处理事件',ledger.incident_key.tolist())
    event=ledger[ledger.incident_key==key].iloc[0]
    with st.form('incident_review'):
        choices=['待复核','已确认','误报','维修中','待复检','已关闭']
        status=st.selectbox('新状态',choices,index=choices.index(event.status))
        note=st.text_area('处理依据 / 维修措施 / 复检结果',key='incident_note')
        st.caption('正常流程：待复核 → 已确认 → 维修中 → 待复检 → 已关闭。误报或关闭后可重新进入待复核。')
        if st.form_submit_button('保存事件处理'):
            try: update_incident(key,status,note);st.rerun()
            except ValueError as exc: st.error(str(exc))
    history=incident_history(key)
    st.dataframe(history,hide_index=True,width='stretch')
    st.download_button('导出完整事件履历',json.dumps({'event':event.to_dict(),'history':history.to_dict('records')},ensure_ascii=False,indent=2), 'incident_history.json','application/json')


def comparison_page(uav_runs):
    st.caption('比较固定数据、任务划分和评估窗口。数据版本或划分不一致时拒绝直接对比。')
    parents=[r for r in uav_runs if not (r/'alert_policy.joblib').exists()]
    if not parents: st.info('请先完成无人机基线训练');return
    with st.expander('创建告警改进版本'):
        parent=st.selectbox('固定基线实验',[r.name for r in parents])
        st.write('按健康训练样本区分低/高角运动代理工况；只用验证集选取阈值、连续确认与解除参数。分类模型和原任务划分保持固定。')
        st.caption('约束：故障任务检出率不低于基线，故障窗口召回率最多下降 5 个百分点；同时考察误报事件数和误报占用时长。不满足条件则保留基线。')
        if st.button('生成并验证告警版本'):
            with st.spinner('拟合工况基线并运行验证集候选对比…'):
                try:
                    output=improve_alerts(ROOT/'runs'/parent)
                    st.success(f'已生成 {output.name}，刷新页面即可选择。')
                except Exception as exc: st.error(str(exc))
        if st.button('刷新版本列表'): st.rerun()
    left=st.selectbox('对照版本',[r.name for r in uav_runs],index=next(i for i,r in enumerate(uav_runs) if r==parents[0]))
    right=st.selectbox('待比较版本',[r.name for r in uav_runs])
    if left==right: st.info('请选择两个不同版本');return
    try: result=compare_runs(ROOT/'runs'/left,ROOT/'runs'/right)
    except ValueError as exc: st.error(str(exc));return
    st.warning('旧测试任务已经查看过，其结果属于回归对比。只有冻结参数后取得的新任务可作为新增任务评价。')
    rows=pd.DataFrame(result['rows'])
    st.dataframe(rows.drop(columns=['delay_reason','detection_delay_s']),hide_index=True,width='stretch')
    st.plotly_chart(px.bar(rows,x='split',y='healthy_alarm_fraction',color='run',barmode='group',title='健康时段告警占比 · 降低误报不能只看事件数'),width='stretch')
    metrics=read_json(ROOT/'runs'/right/'metrics.json')
    if 'alert_improvement' in metrics:
        st.json(metrics['alert_improvement'],expanded=False)
        st.caption('工况是角运动强弱代理，不是已验证的起飞/降落识别。当前数据没有故障起始标注，不报告检测延迟。')
        with st.expander('验证集选参依据'):
            st.dataframe(pd.read_csv(ROOT/'runs'/right/'validation_candidates.csv'),hide_index=True,width='stretch')
    fresh=result['fresh_evaluation']
    if fresh:
        st.subheader('冻结参数后的新增任务评价')
        st.write(f"{fresh['missions']} 个新增任务，{fresh['windows']} 个窗口；五分类 Macro-F1：{fresh['classification']['macro_f1']:.3f}")
        st.dataframe(pd.DataFrame([{'version':'baseline',**fresh['baseline']},{'version':'candidate',**fresh['candidate']}]).drop(columns=['delay_reason','detection_delay_s']),hide_index=True,width='stretch')
        st.caption('来自同一公开平台，尚不能证明跨机体泛化。新任务分数不会用于本次选参。')
    else: st.info('此版本尚未完成新增独立任务评价。')
    st.download_button('下载改进前后报告 HTML',comparison_html(result),'alarm_comparison.html','text/html')
    st.download_button('下载对比指标 JSON',json.dumps(result,ensure_ascii=False,indent=2),'alarm_comparison.json','application/json')
