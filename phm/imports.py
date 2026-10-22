"""CSV inspection, explicit unit mapping, immutable imports and unlabeled inference."""
import csv
import hashlib
import io
import json
from pathlib import Path
import uuid
import joblib
import numpy as np
import pandas as pd
from phm.data import ROOT, CHANNELS, signal_windows, write_json, read_json
from phm.alerts import score_conditioned, confirm_alerts

REQUIRED=['device_id','mission_id','timestamp_s',*CHANNELS]
OPTIONAL=['operating_mode','label']
UNITS={'time':{'s':1.,'ms':.001,'us':.000001},'acc':{'m/s²':1.,'g':9.80665},'gyro':{'rad/s':1.,'deg/s':np.pi/180}}


def read_csv_bytes(payload):
    if len(payload)>50*1024*1024: raise ValueError('CSV 超过 50 MB，请按任务拆分')
    try: text=payload.decode('utf-8-sig')
    except UnicodeDecodeError as exc: raise ValueError('请使用 UTF-8 编码的 CSV') from exc
    header=next(csv.reader(io.StringIO(text)),[])
    if not header or len(set(header))!=len(header): raise ValueError('CSV 表头为空或包含重复列名')
    frame=pd.read_csv(io.StringIO(text),dtype=str,nrows=500001)
    if frame.empty or len(frame)>500000: raise ValueError('要求 1–500000 行数据')
    return frame


def inspect_csv(payload,mapping,units):
    raw=read_csv_bytes(payload)
    selected=[v for v in mapping.values() if v]
    if len(set(selected))!=len(selected): raise ValueError('同一来源列不能映射到多个字段')
    if any(c not in raw for c in selected): raise ValueError('映射引用了不存在的列')
    for kind in UNITS:
        if units.get(kind) not in UNITS[kind]: raise ValueError(f'不支持的 {kind} 单位')
    mapped={k:raw[v] for k,v in mapping.items() if v and k in REQUIRED+OPTIONAL}
    df=pd.DataFrame(mapped,index=raw.index)
    missing=[k for k in REQUIRED if k not in df]
    fatal=[k for k in ['device_id','mission_id','timestamp_s'] if k in missing]
    report={'rows':len(raw),'missing_fields':missing,'blocking_errors':[], 'units':units,
            'warnings':[],'missions':[],'status':'unknown','input_sha256':hashlib.sha256(payload).hexdigest()}
    if fatal:
        report['blocking_errors'].append('必须映射设备、任务和时间字段：'+', '.join(fatal))
        return df,report
    for col in ['device_id','mission_id']:
        if df[col].isna().any() or df[col].fillna('').str.strip().eq('').any():
            report['blocking_errors'].append(f'{col} 含空标识，请补全后导入')
    df['timestamp_s']=pd.to_numeric(df.timestamp_s,errors='coerce')*UNITS['time'][units['time']]
    valid_time=np.isfinite(df.timestamp_s)
    report['invalid_time_rows']=int((~valid_time).sum()); df=df[valid_time].copy()
    if df.empty: report['blocking_errors'].append('没有有效时间戳'); return df,report
    for c in CHANNELS:
        df[c]=pd.to_numeric(df[c],errors='coerce') if c in df else np.nan
        df[c]*=UNITS['acc' if c.startswith('acc_') else 'gyro'][units['acc' if c.startswith('acc_') else 'gyro']]
    df[CHANNELS]=df[CHANNELS].replace([np.inf,-np.inf],np.nan)
    report['invalid_sensor_rows']=int(df[CHANNELS].isna().any(axis=1).sum())
    if 'label' in df:
        df['label']=pd.to_numeric(df.label,errors='coerce')
        if not df.label.isin([0,1,2,3,4]).all(): report['warnings'].append('标签不是完整的 0–4 整数；不用于验证评分')
    df['source_mission_id']=df.mission_id
    df['mission_id']=[json.dumps([str(d),str(m)],ensure_ascii=False) for d,m in zip(df.device_id,df.mission_id)]
    normalized=[]
    for mid,g in df.groupby('mission_id',sort=False):
        g=g.copy(); reversed_steps=int((g.timestamp_s.diff()<0).sum())
        duplicate=g.timestamp_s.duplicated(keep=False)
        # Ambiguous duplicate times cannot silently choose one of conflicting readings.
        duplicates=int(duplicate.sum()); g=g[~duplicate].sort_values('timestamp_s')
        if g.empty:
            report['missions'].append({'mission_id':mid,'rows':0,'duplicate_rows':duplicates,'status':'unknown'}); continue
        span=float(g.timestamp_s.iloc[-1]-g.timestamp_s.iloc[0])
        if span>3600:
            report['blocking_errors'].append(f'{mid} 时长超过 1 小时；请核对时间单位或拆分任务')
        g['original_timestamp_s']=g.timestamp_s
        g['timestamp_s']-=g.timestamp_s.iloc[0]
        valid=np.isfinite(g[CHANNELS]).all(axis=1)
        # Very large magnitudes are retained as missing, never clipped into normal values.
        implausible=(g[CHANNELS].abs()>2000).any(axis=1)
        g.loc[implausible,CHANNELS]=np.nan; valid &= ~implausible
        g['data_quality']=np.where(valid,'valid','unknown')
        report['missions'].append({'mission_id':mid,'device_id':str(g.device_id.iloc[0]),'rows':len(g),
            'duration_s':span,'median_dt_s':float(g.timestamp_s.diff().median()) if len(g)>1 else None,
            'out_of_order_steps':reversed_steps,'duplicate_rows':duplicates,'suspect_unit_rows':int(implausible.sum()),
            'invalid_rows':int((~valid).sum()),'gap_count':int((g.timestamp_s.diff()>.2).sum())})
        normalized.append(g)
    if not normalized: report['blocking_errors'].append('去重后没有可用记录')
    if missing: report['warnings'].append('缺少传感器字段，保存后只能查看质量报告，不能推理')
    report['status']='blocked' if report['blocking_errors'] else ('partial' if missing or report['invalid_sensor_rows'] else 'ready')
    report['warnings'].append('导入数据属于用户提供；模型仅适用于与原训练数据相似的惯性信号，不自动证明跨设备泛化。')
    return pd.concat(normalized,ignore_index=True) if normalized else df.iloc[:0],report


def save_import(payload,mapping,units,name='uploaded.csv',root=None):
    data,report=inspect_csv(payload,mapping,units)
    if report['blocking_errors']: raise ValueError('; '.join(report['blocking_errors']))
    folder=Path(root or ROOT/'data/imports') / f"csv_{report['input_sha256'][:12]}_{uuid.uuid4().hex[:6]}"
    folder.mkdir(parents=True)
    (folder/'input.csv').write_bytes(payload); data.to_parquet(folder/'telemetry.parquet',index=False)
    write_json(folder/'quality.json',report)
    write_json(folder/'manifest.json',{'name':Path(name).name,'data_kind':'user_provided_unverified','mapping':mapping,'units':units,'sha256':report['input_sha256']})
    return folder


def infer_import(folder,run):
    folder,run=Path(folder),Path(run)
    manifest=read_json(folder/'manifest.json'); report=read_json(folder/'quality.json')
    data=pd.read_parquet(folder/'telemetry.parquet'); bundle=joblib.load(run/'models.joblib')
    windows=[]; summaries=[]; quality_windows=[]
    for mid,g in data.groupby('mission_id',sort=False):
        w=signal_windows(g,mid)
        if not w.empty:
            times=np.sort(g.timestamp_s.to_numpy())
            left=np.searchsorted(times,w.start_s.to_numpy())
            right=np.searchsorted(times,w.timestamp_s.to_numpy())-1
            complete=(times[left]-w.start_s.to_numpy()<=.2)&(w.timestamp_s.to_numpy()-times[right]<=.2)
            w=w.loc[complete].copy()
        expected=np.arange(2.,g.timestamp_s.max()+1e-9,1.)
        available=set(w.timestamp_s) if len(w) else set()
        quality_windows.extend({'mission_id':mid,'timestamp_s':float(t),'window_status':'evaluated' if t in available else 'unknown'} for t in expected)
        summaries.append({'mission_id':mid,'usable_windows':len(w),'unknown_windows':len(expected)-len(w),
                          'status':('evaluated' if len(w)==len(expected) else 'partial') if len(w) else 'unknown',
                          'reason':None if len(w) and len(w)==len(expected) else '部分或全部窗口缺少信号、历史不足、采样不足或时间间隔过大'})
        if len(w): windows.append(w)
    out=folder / ('inference_'+run.name); out.mkdir(exist_ok=True)
    pd.DataFrame(quality_windows,columns=['mission_id','timestamp_s','window_status']).to_parquet(out/'window_quality.parquet',index=False)
    if not windows:
        write_json(out/'status.json',{'status':'unknown','missions':summaries,'model_run':run.name,'reason':'没有可用推理窗口；没有输出健康结论'})
        return out
    w=pd.concat(windows,ignore_index=True)
    w['predicted_label']=bundle['classifier'].predict(w[bundle['features']])
    w['class_probability']=bundle['classifier'].predict_proba(w[bundle['features']]).max(axis=1)
    w['anomaly_score']=-bundle['anomaly'].score_samples(w[bundle['features']])
    w['anomaly_threshold']=bundle['anomaly_threshold']; w['raw_anomaly_flag']=w.anomaly_score>w.anomaly_threshold
    policy_path=run/'alert_policy.joblib'
    if policy_path.exists():
        policy=joblib.load(policy_path)
        if policy['conditioned'] is not None: w=score_conditioned(w,policy['conditioned'])
        w,events=confirm_alerts(w,policy['confirm'],policy['release'],policy['merge_gap_s'])
    else: w,events=confirm_alerts(w,1,1,0)
    w.to_parquet(out/'predictions.parquet',index=False); events.to_parquet(out/'alarm_events.parquet',index=False)
    write_json(out/'status.json',{'status':'complete','model_run':run.name,'data_kind':'user_provided_unverified','missions':summaries,
               'source_sha256':manifest['sha256'],'metrics_available':False,'note':'无标签推理不报告准确率；缺失/断流窗口不输出预测'})
    return out
