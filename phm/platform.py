"""Equipment, model bindings, analysis records, alerts and work orders.
SQLite business state is separate from immutable experiment artifacts and legacy reviews.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid
import joblib
import numpy as np
import pandas as pd
from phm.data import ROOT, read_json, sha256, write_json
from phm.training import list_runs
from phm.alerts import score_conditioned, confirm_alerts
from phm.imports import infer_import

DB_PATH=ROOT/'runs/platform.sqlite'
ARTIFACTS_ROOT=ROOT/'runs/analyses'
SCHEMA='''
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY,name TEXT NOT NULL,kind TEXT NOT NULL,serial TEXT NOT NULL UNIQUE,model TEXT,owner TEXT,location TEXT,lifecycle TEXT NOT NULL,provenance TEXT NOT NULL,description TEXT,is_demo INTEGER NOT NULL DEFAULT 0,revision INTEGER NOT NULL DEFAULT 1,created_at TEXT,updated_at TEXT);
CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY,asset_id TEXT NOT NULL REFERENCES assets(id),parent_id TEXT REFERENCES nodes(id),name TEXT NOT NULL,kind TEXT NOT NULL,family TEXT NOT NULL,critical INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS models(run_id TEXT PRIMARY KEY,family TEXT NOT NULL,alias TEXT,enabled INTEGER NOT NULL DEFAULT 1,note TEXT,updated_at TEXT);
CREATE TABLE IF NOT EXISTS bindings(node_id TEXT PRIMARY KEY REFERENCES nodes(id),model_id TEXT NOT NULL REFERENCES models(run_id),source_kind TEXT NOT NULL,source_ref TEXT NOT NULL,case_id TEXT NOT NULL,cutoff REAL NOT NULL DEFAULT 0,updated_at TEXT);
CREATE TABLE IF NOT EXISTS analyses(id TEXT PRIMARY KEY,asset_id TEXT NOT NULL REFERENCES assets(id),revision INTEGER NOT NULL,status TEXT NOT NULL,summary TEXT,created_at TEXT,finished_at TEXT);
CREATE TABLE IF NOT EXISTS results(analysis_id TEXT NOT NULL REFERENCES analyses(id),node_id TEXT NOT NULL REFERENCES nodes(id),payload TEXT NOT NULL,PRIMARY KEY(analysis_id,node_id));
CREATE TABLE IF NOT EXISTS alerts(id TEXT PRIMARY KEY,asset_id TEXT NOT NULL REFERENCES assets(id),node_id TEXT NOT NULL REFERENCES nodes(id),analysis_id TEXT NOT NULL REFERENCES analyses(id),fingerprint TEXT NOT NULL UNIQUE,title TEXT,priority TEXT,status TEXT NOT NULL,evidence TEXT,created_at TEXT,updated_at TEXT);
CREATE TABLE IF NOT EXISTS orders(id TEXT PRIMARY KEY,asset_id TEXT NOT NULL REFERENCES assets(id),alert_id TEXT REFERENCES alerts(id),title TEXT NOT NULL,priority TEXT NOT NULL,assignee TEXT,due_date TEXT,status TEXT NOT NULL,note TEXT,verification TEXT,verification_analysis_id TEXT REFERENCES analyses(id),created_at TEXT,updated_at TEXT);
CREATE TABLE IF NOT EXISTS order_history(id INTEGER PRIMARY KEY,order_id TEXT NOT NULL REFERENCES orders(id),old_status TEXT,new_status TEXT,note TEXT,created_at TEXT);
'''


def now(): return datetime.now(timezone.utc).isoformat()
def uid(prefix): return prefix+'_'+uuid.uuid4().hex[:12]
def encode(obj): return json.dumps(obj,ensure_ascii=False,allow_nan=False)

@contextmanager
def db():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(DB_PATH,timeout=20);con.row_factory=sqlite3.Row
    con.execute('PRAGMA foreign_keys=ON')
    try:
        with con: yield con
    finally: con.close()


def init():
    with db() as con: con.executescript(SCHEMA)


def records(sql,params=()):
    with db() as con: return [dict(r) for r in con.execute(sql,params).fetchall()]


def asset(asset_id):
    found=records('SELECT * FROM assets WHERE id=?',(asset_id,))
    if not found: raise ValueError('装备不存在')
    return found[0]


def create_asset(name,kind,serial,model='',owner='',location='',provenance='用户建档，来源待确认',description='',demo=False):
    if not name.strip() or not serial.strip(): raise ValueError('名称和唯一编号不能为空')
    if kind not in ['UAV','UGV','部件台架']: raise ValueError('不支持的装备类型')
    if not provenance.strip(): raise ValueError('必须说明数据/装备来源')
    aid=uid('EQ');t=now()
    with db() as con:
        try: con.execute('INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(aid,name.strip(),kind,serial.strip(),model,owner,location,'在册',provenance,description,int(demo),1,t,t))
        except sqlite3.IntegrityError as exc: raise ValueError('装备编号已存在') from exc
        templates={'UAV':[('动力','uav',1),('能源','battery',1),('导航','unknown',1),('通信','unknown',1),('执行机构','unknown',1)],
                   'UGV':[('驱动','unknown',1),('能源','battery',1),('转向','unknown',1),('制动','unknown',1),('导航','unknown',0),('通信','unknown',0)],
                   '部件台架':[('试验对象','battery',1)]}
        for title,family,critical in templates[kind]:
            sid=uid('SYS');con.execute('INSERT INTO nodes VALUES (?,?,?,?,?,?,?)',(sid,aid,None,title,'subsystem',family,critical))
            child={'uav':'动力振动监测对象','battery':'电池组件','unknown':title+'组件'}[family]
            con.execute('INSERT INTO nodes VALUES (?,?,?,?,?,?,?)',(uid('CMP'),aid,sid,child,'component',family,critical))
    return aid


def edit_asset(aid,name,model,owner,location,lifecycle,description):
    if not name.strip(): raise ValueError('名称不能为空')
    if lifecycle not in ['在册','维护中','退役']: raise ValueError('无效生命周期')
    asset(aid)
    with db() as con:
        con.execute('UPDATE assets SET name=?,model=?,owner=?,location=?,lifecycle=?,description=?,updated_at=? WHERE id=?',(name.strip(),model,owner,location,lifecycle,description,now(),aid))


def add_component(aid,parent_id,name,family='unknown',critical=False):
    if not name.strip() or family not in ['uav','battery','unknown']: raise ValueError('部件名称或模型类型无效')
    parent=records('SELECT * FROM nodes WHERE id=? AND asset_id=? AND kind=?',(parent_id,aid,'subsystem'))
    if not parent: raise ValueError('分系统必须属于当前装备')
    if asset(aid)['lifecycle']=='退役': raise ValueError('退役装备不能修改结构')
    nid=uid('CMP')
    with db() as con:
        con.execute('INSERT INTO nodes VALUES (?,?,?,?,?,?,?)',(nid,aid,parent_id,name.strip(),'component',family,int(critical)))
        con.execute('UPDATE assets SET revision=revision+1,updated_at=? WHERE id=?',(now(),aid))
    return nid


def model_path(run_id):
    if Path(run_id).name!=run_id: raise ValueError('无效模型编号')
    path=ROOT/'runs'/run_id
    if not (path/'status.json').exists() or read_json(path/'status.json')['status']!='complete': raise ValueError('模型实验不可用')
    return path


def sync_models():
    with db() as con:
        for path in list_runs():
            metrics=read_json(path/'metrics.json');family=metrics['task']
            if family not in ['uav','battery']:continue
            alias=('工况告警 + 螺旋桨分类' if (path/'alert_policy.joblib').exists() else '螺旋桨分类基线') if family=='uav' else '电池 RUL 留一模型'
            con.execute('INSERT OR IGNORE INTO models VALUES (?,?,?,?,?,?)',(path.name,family,alias,1,'公开数据研究模型；绑定前核对适用范围',now()))


def configure_model(run_id,alias,enabled,note):
    model_path(run_id)
    if not alias.strip():raise ValueError('模型名称不能为空')
    with db() as con:
        current=con.execute('SELECT enabled FROM models WHERE run_id=?',(run_id,)).fetchone()
        if not current:raise ValueError('模型尚未注册')
        if current['enabled']!=int(enabled):
            con.execute('UPDATE assets SET revision=revision+1,updated_at=? WHERE id IN (SELECT n.asset_id FROM nodes n JOIN bindings b ON b.node_id=n.id WHERE b.model_id=?)',(now(),run_id))
        con.execute('UPDATE models SET alias=?,enabled=?,note=?,updated_at=? WHERE run_id=?',(alias.strip(),int(enabled),note,now(),run_id))


def sources(family):
    """Read-only catalog. A public case is a demo source, never an equipment identity."""
    result=[]
    runs=[p for p in list_runs() if read_json(p/'metrics.json')['task']==family]
    # Predictions for the same original data recur across models; catalog latest snapshot only.
    if runs:
        path=runs[0]
        if family=='uav':
            w=pd.read_parquet(path/'predictions.parquet')
            for mid,g in w.groupby('mission_id'):
                result.append({'kind':'uav_case','ref':path.name,'case':str(mid),'label':f"TII / {g.split.iloc[0]} / {mid}",'provenance':'公开室内飞行任务；不能作为当前实物装备的实测数据'})
        else:
            curves=pd.read_parquet(path/'curves.parquet')
            for bid in sorted(curves.battery_id.unique()):
                result.append({'kind':'battery_case','ref':path.name,'case':str(bid),'label':f'NASA / {bid}','provenance':'独立实验室电池；不与 TII 飞机拼接为同一真实装备'})
    if family=='uav':
        for p in sorted((ROOT/'data/imports').glob('*/manifest.json')):
            q=read_json(p.parent/'quality.json')
            for mission in q.get('missions',[]):
                if not mission.get('rows'):continue
                result.append({'kind':'csv','ref':p.parent.name,'case':mission['mission_id'],'label':f"CSV / {p.parent.name[-6:]} / {mission['mission_id']}",'provenance':'用户导入，来源与模型适用性由使用者确认'})
    return result


def bind(node_id,run_id,source_kind,source_ref,case_id,cutoff=0):
    node=records('SELECT * FROM nodes WHERE id=? AND kind="component"',(node_id,))
    if not node:raise ValueError('必须绑定到具体部件')
    n=node[0];a=asset(n['asset_id'])
    if a['lifecycle']=='退役':raise ValueError('退役装备不能绑定模型')
    model=records('SELECT * FROM models WHERE run_id=?',(run_id,))
    if not model or not model[0]['enabled'] or model[0]['family']!=n['family']:raise ValueError('模型类型不匹配或已停用')
    if source_kind not in ['uav_case','battery_case','csv']:raise ValueError('未知数据来源')
    if (source_kind=='battery_case')!=(n['family']=='battery'):raise ValueError('数据与部件类型不匹配')
    if not np.isfinite(cutoff) or cutoff<0:raise ValueError('截止时间/循环无效')
    if source_kind=='battery_case' and cutoff!=int(cutoff):raise ValueError('电池截止循环须为整数')
    if source_kind!='csv' and not a['is_demo']:raise ValueError('公开案例只能绑定到演示档案；真实装备请绑定自身导入数据')
    if Path(source_ref).name!=source_ref:raise ValueError('无效来源编号')
    if source_kind=='csv':
        folder=ROOT/'data/imports'/source_ref
        available=pd.read_parquet(folder/'telemetry.parquet').mission_id.unique()
    elif source_kind=='uav_case':available=pd.read_parquet(model_path(source_ref)/'predictions.parquet').mission_id.unique()
    else:available=pd.read_parquet(model_path(source_ref)/'curves.parquet').battery_id.unique()
    if case_id not in available:raise ValueError('数据案例不存在')
    model_path(run_id)
    with db() as con:
        con.execute('INSERT INTO bindings VALUES (?,?,?,?,?,?,?) ON CONFLICT(node_id) DO UPDATE SET model_id=excluded.model_id,source_kind=excluded.source_kind,source_ref=excluded.source_ref,case_id=excluded.case_id,cutoff=excluded.cutoff,updated_at=excluded.updated_at',
                    (node_id,run_id,source_kind,source_ref,case_id,float(cutoff),now()))
        con.execute('UPDATE assets SET revision=revision+1,updated_at=? WHERE id=?',(now(),n['asset_id']))


def unbind(node_id):
    node=records('SELECT asset_id FROM nodes WHERE id=?',(node_id,))
    if not node:raise ValueError('部件不存在')
    with db() as con:
        con.execute('DELETE FROM bindings WHERE node_id=?',(node_id,))
        con.execute('UPDATE assets SET revision=revision+1,updated_at=? WHERE id=?',(now(),node[0]['asset_id']))


def structure(aid):
    return records('SELECT n.*,b.model_id,b.source_kind,b.source_ref,b.case_id,b.cutoff FROM nodes n LEFT JOIN bindings b ON b.node_id=n.id WHERE n.asset_id=? ORDER BY n.rowid',(aid,))


def score_uav(w,run):
    bundle=joblib.load(run/'models.joblib');features=bundle['features']
    if not set(features).issubset(w.columns):raise ValueError('缺少模型需要的特征')
    valid=np.isfinite(w[features]).all(axis=1);w=w[valid].copy()
    if w.empty:raise ValueError('没有可评估的有限值窗口')
    w['predicted_label']=bundle['classifier'].predict(w[features])
    w['class_probability']=bundle['classifier'].predict_proba(w[features]).max(axis=1)
    w['anomaly_score']=-bundle['anomaly'].score_samples(w[features]);w['anomaly_threshold']=bundle['anomaly_threshold']
    w['raw_anomaly_flag']=w.anomaly_score>w.anomaly_threshold
    w['propulsion_hi_proxy']=(100-25*np.maximum(0,(w.acc_norm_std-bundle['health_center'])/bundle['health_scale'])).clip(0,100)
    if (run/'alert_policy.joblib').exists():
        policy=joblib.load(run/'alert_policy.joblib')
        if policy['conditioned'] is not None:w=score_conditioned(w,policy['conditioned'])
        return confirm_alerts(w,policy['confirm'],policy['release'],policy['merge_gap_s'])
    return confirm_alerts(w,1,1,0)


def analyze_component(binding,folder):
    model=records('SELECT * FROM models WHERE run_id=?',(binding['model_id'],))
    if not model or not model[0]['enabled']:raise ValueError('绑定模型已停用')
    run=model_path(binding['model_id']);family=model[0]['family'];cutoff=binding['cutoff'];case=binding['case_id']
    provenance={'model_run':run.name,'source_kind':binding['source_kind'],'source_ref':binding['source_ref'],'case_id':case,'cutoff':cutoff}
    events=[]
    if family=='uav':
        if binding['source_kind']=='csv':
            origin=ROOT/'data/imports'/binding['source_ref'];out=infer_import(origin,run)
            status=read_json(out/'status.json')
            if status['status']!='complete':raise ValueError('导入信号不满足窗口质量要求')
            w=pd.read_parquet(out/'predictions.parquet');w=w[w.mission_id==case]
            provenance['input_sha256']=status['source_sha256']
            if (out/'window_quality.parquet').exists():
                q=pd.read_parquet(out/'window_quality.parquet');q=q[q.mission_id==case]
                if cutoff:q=q[q.timestamp_s<=cutoff]
                provenance['unknown_windows']=int((q.window_status=='unknown').sum())
                provenance['expected_windows']=len(q)
        else:
            source=model_path(binding['source_ref'])/'predictions.parquet'
            w=pd.read_parquet(source);w=w[w.mission_id==case]
            provenance['input_sha256']=sha256(source)
        if cutoff:w=w[w.timestamp_s<=cutoff]
        if w.empty:raise ValueError('截止时间内没有可用窗口')
        # Both published and imported windows are rescored with the explicitly bound model.
        w,detected=score_uav(w,run);w=w.sort_values('timestamp_s')
        w.to_parquet(folder/'trend.parquet',index=False);detected.to_parquet(folder/'events.parquet',index=False)
        latest=w.iloc[-1]
        model_hash=sha256(run/'models.joblib')
        provenance['model_sha256']=model_hash
        if (run/'alert_policy.joblib').exists():provenance['alert_policy_sha256']=sha256(run/'alert_policy.joblib')
        for e in detected.to_dict('records'):
            events.append({'key':e['event_id'],'title':'动力惯性信号持续异常','priority':'中','evidence':e})
        # Classification evidence is kept separate from unsupervised alert events.
        fault_count=int((w.predicted_label!=0).sum());fault_fraction=float((w.predicted_label!=0).mean())
        if not events and fault_count>=3 and fault_fraction>=.1:
            events.append({'key':'classification_review','title':'螺旋桨损伤分类建议复核','priority':'中','evidence':{'fault_windows':fault_count,'window_fraction':fault_fraction,'criterion':'至少 3 个窗口且占比 >=10%，仅复核规则'}})
        result={'family':family,'status':'需复核' if events else '已评估','score':float(latest.propulsion_hi_proxy),'score_label':'动力振动 HI（示意）',
                'latest_class':int(latest.predicted_label),'probability':float(latest.class_probability),'windows':len(w),
                'fault_fraction':fault_fraction,'alert_events':len(events),'rul':None,
                'evidence':'基于惯性信号分类和连续异常；不等于整机安全评估','provenance':provenance}
        if provenance.get('unknown_windows'):
            result['quality']='部分窗口无法评估'
            if not events:result['status']='部分评估'
        return result,events
    source=model_path(binding['source_ref'])/'curves.parquet'
    curves=pd.read_parquet(source);g=curves[curves.battery_id==case].sort_values('cycle_idx')
    if cutoff:g=g[g.cycle_idx<=cutoff]
    if g.empty:raise ValueError('截止循环内没有电池观测')
    current=g.iloc[-1];rul=None;lower=upper=None;reason='历史不足或无适用留一模型'
    fold=run/f'{case}.joblib';provenance['input_sha256']=sha256(source)
    if float(current.capacity_ah)<=1.4:
        rul=0.;reason='当前观测容量已到阈值，非模型外推'
    elif len(g)>=10 and fold.exists():
        b=joblib.load(fold);prefix=g.tail(20)
        slope=float(np.polyfit(prefix.cycle_idx,prefix.capacity_ah,1)[0])
        features=pd.DataFrame([{'cycle_idx':int(current.cycle_idx),'soh':float(current.soh),'slope':slope,'recent_mean':float(prefix.capacity_ah.mean())}])
        rul=max(0.,float(b['model'].predict(features[b['features']])[0]));radius=b['interval_radius'];lower=max(0.,rul-radius);upper=rul+radius
        reason='当前样本留一模型；仅使用截止循环的历史';provenance['model_sha256']=sha256(fold)
    g.to_parquet(folder/'trend.parquet',index=False)
    if float(current.capacity_ah)<=1.4:events.append({'key':'capacity_threshold','title':'电池容量达到试验失效阈值','priority':'高','evidence':{'capacity_ah':float(current.capacity_ah),'threshold_ah':1.4,'cycle_idx':int(current.cycle_idx)}})
    elif rul is not None and rul<=20:events.append({'key':'rul_review','title':'电池预测剩余循环较少，建议容量复检','priority':'中','evidence':{'rul_cycles':rul,'review_threshold_cycles':20,'note':'演示维护规则，非保证寿命'}})
    return {'family':'battery','status':'需复核' if events else '已评估','score':float(current.soh)*100,'score_label':'观测 SOH',
            'capacity_ah':float(current.capacity_ah),'cycle':int(current.cycle_idx),'rul':rul,'lower':lower,'upper':upper,
            'rul_status':reason,'alert_events':len(events),'evidence':'2 Ah 额定容量，1.4 Ah 试验阈值；RUL 单位为放电循环','provenance':provenance},events


def analyze_asset(aid):
    a=asset(aid)
    if a['lifecycle']=='退役':raise ValueError('退役装备不执行新分析')
    nodes=structure(aid);components=[n for n in nodes if n['kind']=='component'];analysis_id=uid('AN')
    folder=ARTIFACTS_ROOT/analysis_id;folder.mkdir(parents=True)
    snapshot={'asset':a,'structure':nodes}
    write_json(folder/'snapshot.json',snapshot)
    with db() as con:con.execute('INSERT INTO analyses VALUES (?,?,?,?,?,?,?)',(analysis_id,aid,a['revision'],'running',None,now(),None))
    results=[];pending=[]
    try:
        for n in components:
            target=folder/n['id'];target.mkdir()
            if not n['model_id']:result={'status':'未评估','score':None,'rul':None,'family':n['family'],'reason':'未绑定适用模型和数据'};events=[]
            else:
                try:result,events=analyze_component(n,target)
                except (ValueError,KeyError,FileNotFoundError,TypeError) as exc:
                    result={'status':'无法评估','score':None,'rul':None,'family':n['family'],'reason':str(exc)};events=[]
            result.update(node_id=n['id'],name=n['name'],parent_id=n['parent_id'])
            results.append(result)
            for e in events:pending.append((n,e))
        evaluated=sum(r['status'] in ['已评估','需复核'] for r in results)
        any_partial=any(r['status']=='部分评估' for r in results)
        summary={'asset_name':a['name'],'component_count':len(components),'evaluated_components':evaluated,
            'coverage':evaluated/len(components) if components else 0.,'health_score':None,
            'overall_status':'需复核' if pending else ('已评估' if evaluated==len(components) and components else '部分评估' if evaluated or any_partial else '未评估'),
            'alert_count':len(pending),'score_reason':'分系统量纲和评分尚未统一标定，整机健康分不计算',
            'source_mode':'公开案例演示' if a['is_demo'] else '用户装备数据','snapshot':snapshot,
            'critical_unknown':[r['name'] for r in results if r['status'] not in ['已评估','需复核'] and next(n['critical'] for n in components if n['id']==r['node_id'])]}
        write_json(folder/'summary.json',summary)
        with db() as con:
            for r in results:con.execute('INSERT INTO results VALUES (?,?,?)',(analysis_id,r['node_id'],encode(r)))
            for n,e in pending:
                identity=[aid,n['id'],n['model_id'],n['source_kind'],n['source_ref'],n['case_id'],e['key']]
                fingerprint=hashlib.sha256(encode(identity).encode()).hexdigest()
                previous=con.execute('SELECT evidence FROM alerts WHERE fingerprint=?',(fingerprint,)).fetchone()
                if previous:
                    e['evidence']['reviews']=json.loads(previous['evidence']).get('reviews',[])
                # Re-analysis updates evidence but never resets review or work-order status.
                con.execute('INSERT INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(fingerprint) DO UPDATE SET analysis_id=excluded.analysis_id,evidence=excluded.evidence,updated_at=excluded.updated_at',
                            (uid('AL'),aid,n['id'],analysis_id,fingerprint,e['title'],e['priority'],'待复核',encode(e['evidence']),now(),now()))
            con.execute('UPDATE analyses SET status=?,summary=?,finished_at=? WHERE id=?',('complete',encode(summary),now(),analysis_id))
        return analysis_id
    except Exception as exc:
        with db() as con:con.execute('UPDATE analyses SET status=?,summary=?,finished_at=? WHERE id=?',('failed',encode({'error':str(exc)}),now(),analysis_id))
        raise


def latest_analysis(aid):
    rows=records('SELECT * FROM analyses WHERE asset_id=? ORDER BY created_at DESC,id DESC LIMIT 1',(aid,))
    if not rows:return None
    row=rows[0];row['summary']=json.loads(row['summary']) if row['summary'] else {}
    row['stale']=row['revision']!=asset(aid)['revision']
    return row


def analysis_results(analysis_id):
    return [json.loads(r['payload']) for r in records('SELECT payload FROM results WHERE analysis_id=?',(analysis_id,))]


def fleet():
    result=[]
    for a in records('SELECT * FROM assets ORDER BY created_at,id'):
        latest=latest_analysis(a['id']);status='未评估';coverage=0
        if latest:
            status='待重新分析' if latest['stale'] else '分析失败' if latest['status']=='failed' else latest['summary'].get('overall_status','分析中')
            if not latest['stale'] and latest['status']=='complete':coverage=latest['summary'].get('coverage',0)
        active=records('SELECT count(*) n FROM alerts WHERE asset_id=? AND status NOT IN ("已关闭","误报")',(a['id'],))[0]['n']
        orders=records('SELECT count(*) n FROM orders WHERE asset_id=? AND status NOT IN ("已关闭","已取消")',(a['id'],))[0]['n']
        result.append({**a,'health_status':status,'coverage':coverage,'active_alerts':active,'open_orders':orders,'last_analysis':latest['created_at'] if latest else None})
    return result


def review_alert(alert_id,status,note):
    if status not in ['已确认','误报']:raise ValueError('仅支持确认或误报')
    if not note.strip():raise ValueError('请填写复核依据')
    with db() as con:
        row=con.execute('SELECT * FROM alerts WHERE id=?',(alert_id,)).fetchone()
        if not row or row['status'] not in ['待复核','已确认','误报']:raise ValueError('告警已进入工单流程，请在工单中处理')
        evidence=json.loads(row['evidence']);evidence.setdefault('reviews',[]).append({'status':status,'note':note,'at':now()})
        con.execute('UPDATE alerts SET status=?,evidence=?,updated_at=? WHERE id=?',(status,encode(evidence),now(),alert_id))


def create_order(aid,title,priority,assignee='',due_date='',note='',alert_id=None):
    asset(aid)
    if not title.strip() or priority not in ['高','中','低']:raise ValueError('工单标题或优先级无效')
    if due_date:
        try:datetime.strptime(due_date,'%Y-%m-%d')
        except ValueError as exc:raise ValueError('计划日期格式应为 YYYY-MM-DD') from exc
    oid=uid('WO');t=now()
    with db() as con:
        if alert_id:
            alert=con.execute('SELECT * FROM alerts WHERE id=? AND asset_id=?',(alert_id,aid)).fetchone()
            if not alert:raise ValueError('告警不属于该装备')
            if alert['status'] in ['误报','已关闭']:raise ValueError('误报或已关闭告警不能直接生成工单')
            existing=con.execute('SELECT id FROM orders WHERE alert_id=? AND status NOT IN ("已关闭","已取消")',(alert_id,)).fetchone()
            if existing:return existing['id']
        con.execute('INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',(oid,aid,alert_id,title.strip(),priority,assignee,due_date,'待处理',note,'',None,t,t))
        con.execute('INSERT INTO order_history(order_id,old_status,new_status,note,created_at) VALUES (?,?,?,?,?)',(oid,None,'待处理',note,t))
        if alert_id:con.execute('UPDATE alerts SET status="处理中",updated_at=? WHERE id=?',(t,alert_id))
    return oid


def update_order(oid,status,note,verification='',verification_analysis_id=None):
    allowed={'待处理':['处理中','已取消'],'处理中':['待复检','已取消'],'待复检':['已关闭','处理中'],'已关闭':[],'已取消':[]}
    if not note.strip():raise ValueError('请填写本次处理记录')
    with db() as con:
        row=con.execute('SELECT * FROM orders WHERE id=?',(oid,)).fetchone()
        if not row:raise ValueError('工单不存在')
        if status not in allowed[row['status']]:raise ValueError('不允许的工单状态跳转')
        if status=='已关闭':
            if not verification.strip():raise ValueError('关闭前须填写复检结果')
            if verification_analysis_id:
                check=con.execute('SELECT * FROM analyses WHERE id=? AND asset_id=? AND status="complete"',(verification_analysis_id,row['asset_id'])).fetchone()
                if not check:raise ValueError('关联复检分析不属于当前装备或未完成')
        t=now()
        con.execute('UPDATE orders SET status=?,note=?,verification=?,verification_analysis_id=?,updated_at=? WHERE id=?',(status,note,verification,verification_analysis_id,t,oid))
        con.execute('INSERT INTO order_history(order_id,old_status,new_status,note,created_at) VALUES (?,?,?,?,?)',(oid,row['status'],status,encode({'note':note,'verification':verification,'analysis':verification_analysis_id}),t))
        if row['alert_id'] and status in ['已关闭','已取消']:
            con.execute('UPDATE alerts SET status=?,updated_at=? WHERE id=?',('已关闭' if status=='已关闭' else '已确认',t,row['alert_id']))


def export_asset(aid):
    return {'asset':asset(aid),'structure':structure(aid),'analyses':records('SELECT * FROM analyses WHERE asset_id=? ORDER BY created_at',(aid,)),
            'alerts':records('SELECT * FROM alerts WHERE asset_id=?',(aid,)),
            'orders':records('SELECT * FROM orders WHERE asset_id=?',(aid,)),
            'order_history':records('SELECT h.* FROM order_history h JOIN orders o ON o.id=h.order_id WHERE o.asset_id=? ORDER BY h.id',(aid,))}


def seed_demo():
    init();sync_models()
    if records('SELECT value FROM settings WHERE key="demo_seeded"'):return
    if records('SELECT id FROM assets'):raise ValueError('已有用户档案，请手动建立演示装备，避免自动混入')
    u=create_asset('多旋翼动力诊断演示','UAV','DEMO-UAV-001',model='公开室内飞行案例',owner='演示组',location='离线实验室',provenance='TII 公开数据；示例档案不代表被接入的实机',demo=True)
    b=create_asset('电池寿命试验台','部件台架','DEMO-BAT-001',model='NASA B0005',owner='算法组',location='退化试验',provenance='NASA 独立电池试验；与无人机案例无装机关系',demo=True)
    create_asset('无人车结构演示','UGV','DEMO-UGV-001',owner='演示组',provenance='仅展示装备层级，无实车数据，不输出健康结论',demo=True)
    models=records('SELECT * FROM models ORDER BY run_id DESC')
    for aid,family in [(u,'uav'),(b,'battery')]:
        available=[m for m in models if m['family']==family and m['enabled']]
        if not available:continue
        choices=sources(family)
        if not choices:continue
        source=next((s for s in choices if ('test' in s['label'] and s['kind']=='uav_case')),choices[0]) if family=='uav' else next((s for s in choices if s['case']=='B0005'),choices[0])
        if family=='uav':
            # Deterministic damaged test example; its role and source remain explicit.
            frame=pd.read_parquet(model_path(source['ref'])/'predictions.parquet')
            damaged=frame[(frame.split=='test')&(frame.label!=0)]
            if len(damaged):source=next(s for s in choices if s['case']==sorted(damaged.mission_id.unique())[0])
        component=next(n for n in structure(aid) if n['kind']=='component' and n['family']==family)
        bind(component['id'],available[0]['run_id'],source['kind'],source['ref'],source['case'],60 if family=='battery' else 0)
    with db() as con:con.execute('INSERT INTO settings VALUES (?,?)',('demo_seeded',now()))
    for aid in [u,b]:analyze_asset(aid)
