"""Real-data PHM validation entrypoint. Does NOT generate synthetic replacements.
Usage:
 python real_data_validation.py rfly --root /path/to/RflyMAD_processed --mapping mapping.json --out results/rfly
 python real_data_validation.py nasa-battery --root /path/to/NASA_mats --out results/battery
The mapping JSON provides native dataset column mappings and fault labels.
"""
import argparse,json,re
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit,LeaveOneGroupOut
from sklearn.metrics import classification_report,confusion_matrix,mean_absolute_error
from scipy.io import loadmat

REQUIRED=['timestamp_s','device_id','fault_label']
def rfly(root,mapping_file,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    mapping=json.loads(Path(mapping_file).read_text(encoding='utf-8'))
    # mapping example: {"columns":{"Time":"timestamp_s","IMU.AccX":"acc_x",...},
    # "features":["acc_x",...],"label_from_parent":true,"id_from_parent":true}
    frames=[];errors=[]
    for f in sorted(root.rglob('*.csv')):
        try:
            df=pd.read_csv(f)
            columns=mapping['columns']
            missing=[k for k in columns if k not in df.columns]
            if missing: errors.append({'file':str(f),'missing_native_columns':missing});continue
            df=df.rename(columns=columns)
            if 'timestamp_s' not in df: raise ValueError('timestamp mapping required')
            if 'fault_label' not in df:
                if not mapping.get('label_from_parent'):raise ValueError('label source required')
                df['fault_label']=f.parent.name
            if 'device_id' not in df:
                if not mapping.get('id_from_parent'):raise ValueError('flight/case identifier required')
                df['device_id']=f.parent.as_posix()
            df['source_file']=str(f)
            frames.append(df[['timestamp_s','device_id','fault_label','source_file']+mapping['features']].copy())
        except Exception as e:errors.append({'file':str(f),'error':str(e)})
    (out/'ingestion_report.json').write_text(json.dumps({'files_loaded':len(frames),'errors':errors},ensure_ascii=False,indent=2),encoding='utf-8')
    if not frames:raise RuntimeError('No usable CSV input; obtain genuine processed RflyMAD files and adapt mapping.json. No metrics generated.')
    df=pd.concat(frames,ignore_index=True)
    df['timestamp_s']=pd.to_numeric(df.timestamp_s,errors='coerce')
    for c in mapping['features']:df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['timestamp_s'])
    rows=[]
    for flight,g in df.groupby('device_id'):
        g=g.sort_values('timestamp_s')
        for end in range(64,len(g)+1,32):
            w=g.iloc[end-64:end];row={'device_id':flight,'fault_label':w.fault_label.iloc[-1]}
            for c in mapping['features']:
                v=w[c].to_numpy(float);row[c+'_mean']=np.nanmean(v) if np.isfinite(v).any() else np.nan;row[c+'_std']=np.nanstd(v) if np.isfinite(v).any() else np.nan
            rows.append(row)
    w=pd.DataFrame(rows)
    if len(w)<10 or w.device_id.nunique()<4 or w.fault_label.nunique()<2:raise RuntimeError('Insufficient distinct flights/classes for reliable grouped holdout. No metric issued.')
    feats=[c for c in w if c.endswith(('_mean','_std'))]
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42).split(w,groups=w.device_id))
    if w.iloc[tr].fault_label.nunique()<2:raise RuntimeError('Training split contains less than 2 classes; expand dataset.')
    model=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=250,min_samples_leaf=2,class_weight='balanced',random_state=42))
    model.fit(w.iloc[tr][feats],w.iloc[tr].fault_label);pred=model.predict(w.iloc[te][feats]);actual=w.iloc[te].fault_label
    summary={'validation_status':'REAL_DATA_EVALUATED','data_source':'RflyMAD user-provided processed CSV','flights_total':int(w.device_id.nunique()),'flights_test':int(w.iloc[te].device_id.nunique()),'windows_test':len(te),'report':classification_report(actual,pred,zero_division=0,output_dict=True),'labels':sorted(set(actual)|set(pred))}
    summary['confusion_matrix']=confusion_matrix(actual,pred,labels=summary['labels']).tolist()
    (out/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame({'flight':w.iloc[te].device_id,'true':actual,'pred':pred}).to_csv(out/'predictions.csv',index=False)
    print(json.dumps({'validation_status':summary['validation_status'],'macro_f1':summary['report']['macro avg']['f1-score'],'flights_test':summary['flights_test']},ensure_ascii=False))

def mat_discharge_capacities(path):
    # NASA B0005/B0006/B0007/B0018 MATLAB struct format (v5); file names and nested fields verified at load time.
    raw=loadmat(path,squeeze_me=True,struct_as_record=False)
    k=path.stem
    if k not in raw:raise ValueError(f'MAT file lacks root {k}')
    data=raw[k];cycles=np.atleast_1d(data.cycle)
    ans=[]
    for i,cycle in enumerate(cycles):
        if str(cycle.type).strip().lower()!='discharge':continue
        try:
            cap=float(np.asarray(cycle.data.Capacity).squeeze())
            if np.isfinite(cap) and cap>0:ans.append((i,cap))
        except (AttributeError,ValueError,TypeError):pass
    return pd.DataFrame(ans,columns=['source_cycle','capacity_ah'])

def nasa_battery(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths=sorted(root.rglob('B*.mat'))
    if not paths:raise FileNotFoundError('NASA battery .mat files not present. Download from official NASA PCoE; no synthetic fallback.')
    series=[]
    for p in paths:
        try:
            a=mat_discharge_capacities(p)
            if len(a)<20:continue
            a['battery_id']=p.stem;a['cycle_idx']=np.arange(len(a));a['soh']=a.capacity_ah/a.capacity_ah.iloc[0];series.append(a)
        except Exception as e:print('Skip',p,e)
    if len(series)<3:raise RuntimeError('Require >=3 battery specimens for specimen-held-out evaluation; no metric generated.')
    all_data=pd.concat(series,ignore_index=True);all_data.to_csv(out/'capacity_curves.csv',index=False)
    # Each specimen's EOL is a first observed threshold crossing; exclude censored cells.
    eol={}
    for bid,g in all_data.groupby('battery_id'):
        hit=g[g.soh<=.7]
        if len(hit):eol[bid]=int(hit.cycle_idx.iloc[0])
    if len(eol)<3:raise RuntimeError('Fewer than three specimens reach 70% initial-capacity threshold; EOL cannot be validated as defined.')
    sub=all_data[all_data.battery_id.isin(eol)].copy()
    sub['rul_cycles']=sub.apply(lambda r:max(0,eol[r.battery_id]-r.cycle_idx),axis=1)
    # do not leak future specimen capacity; model uses observed capacity+cycle only
    groups=sub.battery_id.unique();preds=[]
    for held in groups:
        train=sub[sub.battery_id!=held];test=sub[sub.battery_id==held]
        from sklearn.ensemble import RandomForestRegressor
        reg=RandomForestRegressor(n_estimators=250,min_samples_leaf=3,random_state=42)
        reg.fit(train[['cycle_idx','soh']],train.rul_cycles)
        x=reg.predict(test[['cycle_idx','soh']]);preds.append(pd.DataFrame({'battery_id':held,'cycle_idx':test.cycle_idx,'true_rul':test.rul_cycles,'pred_rul':np.maximum(x,0)}))
    predictions=pd.concat(preds,ignore_index=True);predictions.to_csv(out/'rul_predictions.csv',index=False)
    per={bid:float(mean_absolute_error(g.true_rul,g.pred_rul)) for bid,g in predictions.groupby('battery_id')}
    metrics={'validation_status':'REAL_DATA_EVALUATED','data_source':'NASA PCoE battery MATLAB','evaluation':'leave-one-battery-out','eol_definition':'first observed SOH <= 0.70 of first discharge capacity','batteries':list(eol),'eol_cycle_by_battery':eol,'mae_cycles':float(mean_absolute_error(predictions.true_rul,predictions.pred_rul)),'mae_by_battery':per,'limitations':'Only specimens with observed EOL used; first-cycle capacity normalization is not rated-capacity SOH; RUL is per discharge cycle, not flight hours.'}
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(metrics,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='task',required=True)
    a=s.add_parser('rfly');a.add_argument('--root',required=True);a.add_argument('--mapping',required=True);a.add_argument('--out',default='results/rfly')
    b=s.add_parser('nasa-battery');b.add_argument('--root',required=True);b.add_argument('--out',default='results/nasa_battery')
    x=p.parse_args();rfly(x.root,x.mapping,x.out) if x.task=='rfly' else nasa_battery(x.root,x.out)
