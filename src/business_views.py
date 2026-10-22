"""Business workspaces built on persisted equipment and analysis evidence."""
import json
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
from phm import platform as p
from src.lab_views import render_lab, chart, metric_row


def perform(fn,*args,**kwargs):
    try:
        result=fn(*args,**kwargs)
        st.session_state['notice']='操作已保存'+(f' · {result}' if isinstance(result,str) else '')
        st.rerun()
    except (ValueError,FileNotFoundError,KeyError,sqlite3.Error) as exc:st.error(str(exc))


def frame(rows,columns=None):
    if not rows:st.info('暂无记录。');return
    df=pd.DataFrame(rows)
    if 'family' in df:df['family']=df['family'].map({'uav':'动力诊断','battery':'电池寿命','unknown':'待接入','servo':'伺服诊断'})
    for flag in ['critical','enabled']:
        if flag in df:df[flag]=df[flag].map({0:'否',1:'是'})
    if columns:df=df[list(columns)].rename(columns=columns)
    st.dataframe(df,hide_index=True,width='stretch')


def pick_asset(key):
    assets=p.records('SELECT * FROM assets ORDER BY created_at,id')
    if not assets:st.info('请先到装备管理建档，或在监测中心初始化公开数据演示。');return None
    index=next((i for i,a in enumerate(assets) if a['id']==st.session_state.get('active_asset')),0)
    aid=st.selectbox('当前装备',[a['id'] for a in assets],index=index,format_func=lambda aid:next(a['name']+' · '+a['serial'] for a in assets if a['id']==aid),key=key)
    st.session_state['active_asset']=aid
    a=p.asset(aid)
    st.caption(('公开案例演示档案' if a['is_demo'] else '用户装备档案')+' · '+a['provenance'])
    return a


def monitor(view="装备态势"):
    rows=p.fleet()
    if not rows:
        st.info('开始演示：建立无人机、电池台架与无人车示例档案，并运行已有真实数据模型。')
        if st.button('初始化公开数据演示',type='primary'):perform(p.seed_demo)
        return
    if view=='装备态势':
        metric_row([('在册装备',str(sum(a['lifecycle']!='退役' for a in rows)),'含明确标注的演示档案'),('需复核装备',str(sum(a['active_alerts']>0 for a in rows)),'存在未关闭告警'),('待办工单',str(sum(a['open_orders'] for a in rows)),'未关闭且未取消'),('未完整评估',str(sum(a['coverage']<1 for a in rows)),'未接入与过期结论不视为健康')])
        left,right=st.columns([2,1])
        with left:
            st.subheader('装备态势')
            kind=st.selectbox('装备类型',['全部','UAV','UGV','部件台架','伺服台架'])
            shown=[a for a in rows if kind=='全部' or a['kind']==kind]
            frame([{**a,'coverage':f"{a['coverage']:.0%}",'mode':'演示' if a['is_demo'] else '用户'} for a in shown],{'name':'装备','kind':'类型','mode':'来源','health_status':'评估状态','coverage':'部件覆盖','active_alerts':'告警','open_orders':'待办'})
        with right:
            st.subheader('评估覆盖')
            for a in rows:
                st.progress(float(a['coverage']),text=a['name']+f" · {a['coverage']:.0%}")
            st.caption('已评估部件 / 全部部件')
        st.caption('离线分析快照，非实时遥测。整机健康分尚未标定；覆盖率表示有有效结论的部件比例，不代表安全概率。')
        return
    st.subheader('异常与处置')
    alerts=p.records('SELECT al.*,a.name asset_name FROM alerts al JOIN assets a ON a.id=al.asset_id ORDER BY al.updated_at DESC')
    status=st.selectbox('告警范围',['待处理','全部','已关闭','误报'])
    alerts=[a for a in alerts if status=='全部' or (status=='待处理' and a['status'] not in ['已关闭','误报']) or a['status']==status]
    frame(alerts,{'asset_name':'装备','title':'异常','priority':'优先级','status':'状态','updated_at':'更新时间'})
    if alerts:
        selected=st.selectbox('查看告警',[a['id'] for a in alerts],format_func=lambda x:next(a['asset_name']+' / '+a['title'] for a in alerts if a['id']==x))
        a=next(a for a in alerts if a['id']==selected)
        with st.expander('分析证据与来源',expanded=False):
            st.caption('分析快照 '+a['analysis_id']);st.json(json.loads(a['evidence']))
            result=next((r for r in p.analysis_results(a['analysis_id']) if r['node_id']==a['node_id']),{})
            st.json(result.get('provenance',{}))
        if a['status'] in ['待复核','已确认','误报']:
            with st.form('alert_review'):
                outcome=st.selectbox('复核结论',['已确认','误报']);note=st.text_area('复核依据')
                if st.form_submit_button('保存复核'):perform(p.review_alert,selected,outcome,note)
        if a['status'] not in ['已关闭','误报']:
            with st.form('alert_order'):
                c1,c2=st.columns(2);owner=c1.text_input('工单负责人');due=c2.text_input('计划日期',placeholder='YYYY-MM-DD')
                if st.form_submit_button('生成维修工单'):perform(p.create_order,a['asset_id'],a['title'],a['priority'],owner,due,'由告警生成，请核对模型证据',selected)


def equipment(view='装备档案'):
    if view=='新增装备':
        with st.form('asset_new'):
            c1,c2=st.columns(2);name=c1.text_input('装备名称');serial=c2.text_input('唯一编号')
            kind=c1.selectbox('装备类型',['UAV','UGV','部件台架','伺服台架']);model=c2.text_input('型号')
            owner=c1.text_input('责任人');location=c2.text_input('位置 / 单位')
            demo=st.checkbox('演示档案（允许绑定公开案例）');source=st.text_input('来源说明',value='用户建档，数据来源待确认')
            if st.form_submit_button('创建装备',type='primary'):perform(p.create_asset,name,kind,serial,model,owner,location,source,demo=demo)
        return
    a=pick_asset('equipment_asset')
    if not a:return
    if view=='装备档案':
        with st.form('asset_edit'):
            c1,c2=st.columns(2);name=c1.text_input('名称',a['name']);model=c2.text_input('型号',a['model'])
            owner=c1.text_input('负责人',a['owner']);location=c2.text_input('位置',a['location'])
            state=st.selectbox('生命周期',['在册','维护中','退役'],index=['在册','维护中','退役'].index(a['lifecycle']))
            description=st.text_area('装备说明',a['description'])
            if st.form_submit_button('保存档案'):perform(p.edit_asset,a['id'],name,model,owner,location,state,description)
        st.download_button('导出装备全履历 JSON',p.encode(p.export_asset(a['id'])),file_name=a['serial']+'_history.json',mime='application/json')
    nodes=p.structure(a['id']);systems=[n for n in nodes if n['kind']=='subsystem'];components=[n for n in nodes if n['kind']=='component']
    if view=='分系统与部件':
        for system in systems:
            with st.expander(system['name']+'分系统',expanded=True):
                frame([n for n in components if n['parent_id']==system['id']],{'name':'部件','family':'模型领域','critical':'关键部件','model_id':'绑定版本','case_id':'数据案例'})
        with st.form('component_add'):
            parent=st.selectbox('所属分系统',[n['id'] for n in systems],format_func=lambda x:next(n['name'] for n in systems if n['id']==x))
            name=st.text_input('部件名称');family=st.selectbox('评估领域',['unknown','uav','battery','servo'],format_func=lambda x:{'unknown':'尚无模型','uav':'动力惯性信号','battery':'电池退化','servo':'伺服振动'}[x]);critical=st.checkbox('关键部件')
            if st.form_submit_button('增加部件'):perform(p.add_component,a['id'],parent,name,family,critical)
    if view=='数据与模型绑定':
        st.caption('每个部件绑定明确的数据案例和模型版本。变更绑定后，旧分析会标为待重新分析。0 表示使用全部已导入观测。')
        nid=st.selectbox('待绑定部件',[n['id'] for n in components],format_func=lambda x:next(n['name'] for n in components if n['id']==x))
        n=next(n for n in components if n['id']==nid)
        models=p.records('SELECT * FROM models WHERE family=? AND enabled=1 ORDER BY run_id DESC',(n['family'],))
        sources=p.sources(n['family']) if n['family']!='unknown' else []
        sources=[s for s in sources if a['is_demo'] or s['kind']=='csv']
        if models and sources:
            model=st.selectbox('模型版本',[m['run_id'] for m in models],index=next((i for i,m in enumerate(models) if m['run_id']==n['model_id']),0),format_func=lambda x:next(m['alias']+' / '+x for m in models if m['run_id']==x))
            ix=st.selectbox('数据案例',list(range(len(sources))),index=next((i for i,s in enumerate(sources) if s['ref']==n['source_ref'] and s['case']==n['case_id']),0),format_func=lambda i:sources[i]['label'])
            cutoff=st.number_input('分析截止循环' if n['family']=='battery' else '分析截止秒',min_value=0.,value=float(n['cutoff'] or 0),step=1.)
            st.caption(sources[ix]['provenance'])
            if st.button('保存数据与模型绑定',type='primary'):
                s=sources[ix];perform(p.bind,nid,model,s['kind'],s['ref'],s['case'],cutoff)
        else:st.info('暂无兼容的模型或数据。可到健康分析导入飞行 CSV，或在模型中心完成训练；其他分系统需要相应数据与模型。')
        if n['model_id'] and st.button('解除当前绑定'):perform(p.unbind,nid)


def health(uav_run,battery_run,view='装备健康分析'):
    if view=='伺服执行机构诊断':
        from src.servo_views import case_page
        case_page();return
    if view!='装备健康分析':render_lab(view,uav_run,battery_run);return
    a=pick_asset('health_asset')
    if not a:return
    if st.button('运行装备健康分析',type='primary',disabled=a['lifecycle']=='退役'):
        with st.spinner('按部件绑定的模型与截止观测生成分析快照…'):perform(p.analyze_asset,a['id'])
    latest=p.latest_analysis(a['id'])
    if not latest:st.info('绑定数据和模型后运行分析，即可查看诊断、趋势和寿命。');return
    if latest['stale']:st.warning('装备结构或模型绑定已改变。以下为历史结果，请重新分析。')
    history=p.records('SELECT * FROM analyses WHERE asset_id=? ORDER BY created_at DESC,id DESC',(a['id'],))
    anid=st.selectbox('分析记录',[r['id'] for r in history],format_func=lambda x:next(r['created_at'][:19]+' / '+r['status']+' / '+x for r in history if r['id']==x))
    run=next(r for r in history if r['id']==anid);summary=json.loads(run['summary'] or '{}')
    if run['status']!='complete':st.warning('该次分析未完成：'+str(summary));return
    metric_row([('综合评估',summary['overall_status'],'按部件结论汇总'),('有效覆盖',f"{summary['coverage']:.0%}",'可评估部件数 / 全部部件数'),('异常证据',str(summary['alert_count']),'本次分析检出的事件'),('整机健康分','未标定',summary['score_reason'])])
    if summary['critical_unknown']:st.info('关键部件尚未评估：'+'、'.join(summary['critical_unknown']))
    results=sorted(p.analysis_results(anid),key=lambda r:r.get("score") is None)
    frame([{**r,'score':f"{r['score']:.1f}" if r.get('score') is not None else '—','rul':f"{r['rul']:.1f}" if r.get('rul') is not None else '—'} for r in results],{'name':'部件','family':'领域','status':'状态','score':'局部指标','rul':'剩余循环'})
    selected=st.selectbox('部件分析详情',[r['node_id'] for r in results],format_func=lambda x:next(r['name'] for r in results if r['node_id']==x))
    r=next(r for r in results if r['node_id']==selected)
    if r.get('reason'):st.info(r['reason'])
    else:
        st.caption(r['evidence'])
        if r['family']=='uav':
            st.write(f"最新分类：{r['latest_class']} 个损伤螺旋桨类别；分类概率 {r['probability']:.1%}；故障分类窗口占比 {r['fault_fraction']:.1%}。")
        elif r['family']=='servo':
            st.write(f"最新状态：{r['class_name']}；分类概率 {r['probability']:.1%}。")
            st.caption(r['recommended_action'])
            if 'reference_class_name' in r:
                st.caption('公开案例标注：'+r['reference_class_name']+'（仅事后对照，未参与模型输入）')
                if r['latest_class']!=r['reference_class']:st.warning('当前有效窗口与案例标注不一致，是已知误判，须人工复核。')
        else:
            st.write(f"观测 SOH {r['score']:.1f}%，容量 {r['capacity_ah']:.3f} Ah。{r['rul_status']}")
            if r['rul'] is not None:
                st.write(f"预测剩余循环：{r['rul']:.1f}"+(f"；经验误差区间 {r['lower']:.1f}–{r['upper']:.1f}" if r['lower'] is not None else ''))
        path=p.ARTIFACTS_ROOT/anid/selected/'trend.parquet'
        if path.exists():
            df=pd.read_parquet(path)
            fields=['propulsion_hi_proxy','anomaly_score','class_probability'] if r['family']=='uav' else ['capacity_ah','soh']
            if r['family']=='servo':fields=['rms','crest','kurtosis','dominant_hz','class_probability']
            field=st.selectbox('趋势指标',fields)
            chart(px.line(df,x='cycle_idx' if r['family']=='battery' else 'timestamp_s',y=field))
        with st.expander('来源、模型版本与数据质量'):st.json(r['provenance'])
    st.download_button('导出本次分析',p.encode({'analysis':run,'results':results}),file_name=anid+'.json',mime='application/json')


def maintenance(uav_run,battery_run,view='工单与履历'):
    if view!='工单与履历':render_lab(view,uav_run,battery_run);return
    a=pick_asset('maintenance_asset')
    if not a:return
    rows=p.records('SELECT * FROM orders WHERE asset_id=? ORDER BY created_at DESC',(a['id'],))
    metric_row([('待处理',str(sum(o['status']=='待处理' for o in rows)),'待安排维修'),('处理中',str(sum(o['status']=='处理中' for o in rows)),'已开始维修'),('待复检',str(sum(o['status']=='待复检' for o in rows)),'需记录复检结果'),('已关闭',str(sum(o['status']=='已关闭' for o in rows)),'已完成维修闭环')])
    with st.expander('新建维修工单'):
        with st.form('new_order'):
            title=st.text_input('维修任务');priority=st.selectbox('优先级',['中','高','低']);assignee=st.text_input('负责人');due=st.text_input('计划日期',placeholder='YYYY-MM-DD');note=st.text_area('任务说明')
            if st.form_submit_button('创建工单'):perform(p.create_order,a['id'],title,priority,assignee,due,note)
    frame(rows,{'id':'工单号','title':'维修任务','priority':'优先级','assignee':'负责人','due_date':'计划日期','status':'状态'})
    if rows:
        oid=st.selectbox('工单详情',[o['id'] for o in rows],format_func=lambda x:next(o['title']+' / '+o['status'] for o in rows if o['id']==x));o=next(o for o in rows if o['id']==oid)
        st.caption('关联告警：'+str(o['alert_id'] or '人工创建'));st.write(o['note'])
        allowed={'待处理':['处理中','已取消'],'处理中':['待复检','已取消'],'待复检':['已关闭','处理中'],'已关闭':[],'已取消':[]}
        if allowed[o['status']]:
            with st.form('order_update'):
                status=st.selectbox('下一状态',allowed[o['status']]);note=st.text_area('处理记录');verification=st.text_area('复检结果（关闭必填）')
                analyses=p.records('SELECT id FROM analyses WHERE asset_id=? AND status="complete" ORDER BY created_at DESC',(a['id'],))
                analysis=st.selectbox('关联复检分析（可选）',['']+[x['id'] for x in analyses],format_func=lambda x:x or '人工复检，无模型分析')
                if st.form_submit_button('保存工单进展',type='primary'):perform(p.update_order,oid,status,note,verification,analysis or None)
        frame(p.records('SELECT * FROM order_history WHERE order_id=? ORDER BY id',(oid,)),{'created_at':'记录时间','old_status':'原状态','new_status':'新状态','note':'处理依据'})
    st.download_button('导出维修履历',p.encode(p.export_asset(a['id'])),file_name=a['serial']+'_maintenance.json',mime='application/json')


def models(uav_run,battery_run,view='模型目录与绑定'):
    if view=='伺服训练与验证':
        from src.servo_views import training_page
        training_page();return
    if view!='模型目录与绑定':render_lab(view,uav_run,battery_run);return
    if st.button('同步已完成实验'):perform(p.sync_models)
    rows=p.records('SELECT * FROM models ORDER BY run_id DESC')
    frame(rows,{'alias':'模型名称','family':'任务','run_id':'版本','enabled':'可绑定','note':'适用说明'})
    if not rows:return
    mid=st.selectbox('管理模型',[m['run_id'] for m in rows]);m=next(m for m in rows if m['run_id']==mid)
    path=p.model_path(mid);metrics=p.read_json(path/'metrics.json')
    if m['family']=='uav':
        test=metrics['results']['test'];metric_row([('测试 Macro-F1',f"{test['macro_f1']:.3f}",'按任务划分的测试窗口'),('测试准确率',f"{test['accuracy']:.3f}",'不是整机故障检出率')])
    elif m['family']=='servo':
        test=metrics['results']['test'];metric_row([('记录 Macro-F1',f"{test['records']['macro_f1']:.3f}",'按转速隔离的完整记录'),('窗口 Macro-F1',f"{test['windows']['macro_f1']:.3f}",'1 秒非重叠窗口')])
    else:
        metric_row([('RUL MAE',f"{metrics['random_forest']['mae']:.2f}",'循环，留一电池验证'),('RUL RMSE',f"{metrics['random_forest']['rmse']:.2f}",'循环，失效前评估')])
    with st.form('model_registry'):
        alias=st.text_input('展示名称',m['alias']);enabled=st.checkbox('允许绑定与新分析',bool(m['enabled']));note=st.text_area('适用范围 / 限制',m['note'])
        if st.form_submit_button('保存模型配置'):perform(p.configure_model,mid,alias,enabled,note)
    st.subheader('当前应用关系')
    frame(p.records('SELECT a.name asset_name,n.name component,b.case_id FROM bindings b JOIN nodes n ON n.id=b.node_id JOIN assets a ON a.id=n.asset_id WHERE b.model_id=?',(mid,)),{'asset_name':'装备','component':'部件','case_id':'案例'})
    with st.expander('完整训练与验证指标'):st.json(metrics)
    st.download_button('导出验证指标',p.encode(metrics),file_name=mid+'_metrics.json',mime='application/json')
