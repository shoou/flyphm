"""UAV/UGV PHM baseline. Synthetic data only; not safety-certified."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

FEATURES = ['motor_current_a','motor_temp_c','vibration_rms','tracking_error','battery_voltage_v','battery_temp_c','nav_residual','comm_loss_pct']
FAULTS = ['healthy','motor','battery','actuator','navigation','communication']
SUBSYSTEMS = {'propulsion':['motor_current_a','motor_temp_c','vibration_rms'], 'energy':['battery_voltage_v','battery_temp_c'], 'actuator':['tracking_error'], 'navigation':['nav_residual'], 'communication':['comm_loss_pct']}
# Illustrative engineering reference ranges. Replace by platform- and operating-mode-specific baselines.
RANGES = {'motor_current_a':(5,35),'motor_temp_c':(25,85),'vibration_rms':(0,3.0),'tracking_error':(0,.25),'battery_voltage_v':(40,48),'battery_temp_c':(20,60),'nav_residual':(0,2.5),'comm_loss_pct':(0,5)}
WEIGHTS = {'propulsion':.32,'energy':.25,'actuator':.18,'navigation':.15,'communication':.10}
CRITICAL = ['propulsion','energy','actuator']

def make_data(n_devices=36, steps=90, seed=42):
    rng = np.random.default_rng(seed)
    frames=[]
    for d in range(n_devices):
        device_type = 'UAV' if d%2 == 0 else 'UGV'
        fault = FAULTS[d % len(FAULTS)]
        t=np.arange(steps)
        load=rng.uniform(.15,.9,size=steps)
        baseline={
            'motor_current_a': 8+15*load+rng.normal(0,1.4,steps),
            'motor_temp_c':38+15*load+rng.normal(0,2,steps),
            'vibration_rms':.65+.65*load+rng.normal(0,.12,steps),
            'tracking_error':.025+.045*load+rng.normal(0,.008,steps),
            'battery_voltage_v':47.5-2.1*load-rng.normal(0,.30,steps),
            'battery_temp_c':28+12*load+rng.normal(0,1.4,steps),
            'nav_residual':.28+.2*load+rng.normal(0,.09,steps),
            'comm_loss_pct':.15+.25*load+rng.normal(0,.08,steps),
        }
        if fault != 'healthy':
            severity=np.clip((t-30)/(steps-30),0,1)
            if fault=='motor':
                baseline['vibration_rms']+=2.4*severity
                baseline['motor_temp_c']+=24*severity
                baseline['motor_current_a']+=10*severity
            elif fault=='battery':
                baseline['battery_voltage_v']-=6*severity
                baseline['battery_temp_c']+=20*severity
            elif fault=='actuator': baseline['tracking_error']+=.35*severity
            elif fault=='navigation': baseline['nav_residual']+=3.2*severity
            elif fault=='communication': baseline['comm_loss_pct']+=9*severity
        df=pd.DataFrame(baseline)
        df.insert(0,'timestamp_s',t.astype(float))
        df.insert(0,'device_type',device_type)
        df.insert(0,'device_id',f'D{d:03d}')
        df['fault_label'] = np.where((t>=45)&(fault!='healthy'),fault,'healthy')
        frames.append(df)
    return pd.concat(frames,ignore_index=True)

def make_windows(df, window=12, stride=6):
    out=[]
    for device_id,g in df.groupby('device_id'):
        g=g.sort_values('timestamp_s')
        for stop in range(window,len(g)+1,stride):
            sl=g.iloc[stop-window:stop]
            row={'device_id':device_id, 'device_type':sl.device_type.iloc[-1],
                 'timestamp_s':float(sl.timestamp_s.iloc[-1]),'fault_label':sl.fault_label.iloc[-1]}
            for f in FEATURES:
                v=sl[f].to_numpy(dtype=float)
                row[f]=float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                row[f+'_std']=float(np.nanstd(v)) if np.isfinite(v).any() else np.nan
            out.append(row)
    return pd.DataFrame(out)

def train(df, outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    windows=make_windows(df)
    cols=[c for c in windows if c in FEATURES or c.endswith('_std')]
    split=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42)
    tr,te=next(split.split(windows,windows.fault_label,groups=windows.device_id))
    train_df,test_df=windows.iloc[tr],windows.iloc[te]
    model=Pipeline([('imputer',SimpleImputer(strategy='median')),
                    ('classifier',RandomForestClassifier(n_estimators=180,min_samples_leaf=2,class_weight='balanced',random_state=42))])
    model.fit(train_df[cols],train_df.fault_label)
    pred=model.predict(test_df[cols])
    report=classification_report(test_df.fault_label,pred,zero_division=0,output_dict=True)
    # Train anomaly detector only on known healthy windows; device-level split is retained.
    healthy=train_df[train_df.fault_label=='healthy']
    anomaly=Pipeline([('imputer',SimpleImputer(strategy='median')),
                      ('detector',IsolationForest(n_estimators=150,contamination=.05,random_state=42))])
    anomaly.fit(healthy[cols])
    joblib.dump({'model':model,'anomaly':anomaly,'features':cols},outdir/'models.joblib')
    (outdir/'metrics.json').write_text(json.dumps({'test_devices':sorted(test_df.device_id.unique().tolist()),'classification':report,'labels':sorted(set(test_df.fault_label)|set(pred)),'confusion_matrix':confusion_matrix(test_df.fault_label,pred,labels=sorted(set(test_df.fault_label)|set(pred))).tolist()},ensure_ascii=False,indent=2),encoding='utf-8')
    return {'windows':len(windows),'test_devices':test_df.device_id.nunique(),'macro_f1':round(report['macro avg']['f1-score'],3)}

def health_scores(row):
    scores={}
    for name,cols in SUBSYSTEMS.items():
        penalties=[]
        for field in cols:
            x=float(row[field]); low,high=RANGES[field]
            # mid-point/upper/lower directional rules are deliberately illustrative.
            if field=='battery_voltage_v': p=np.clip((high-x)/(high-low),0,1)
            else: p=np.clip((x-low)/(high-low),0,1)
            penalties.append(float(p))
        scores[name]=round(100*(1-np.mean(penalties)),1)
    base=sum(WEIGHTS[k]*scores[k] for k in scores)
    overall=min(base,min(scores[k]+10 for k in CRITICAL))
    return {'overall':round(max(0,min(100,overall)),1),'subsystems':scores,'score_status':'illustrative_uncalibrated'}

def rul_from_history(history, feature='vibration_rms', threshold=3.0, min_points=12):
    """Demonstration of threshold-crossing horizon, not a validated RUL predictor."""
    h=history[['timestamp_s',feature]].dropna().sort_values('timestamp_s').tail(35)
    if len(h)<min_points or h.timestamp_s.nunique()<min_points:
        return {'status':'insufficient_history','rul_hours':None}
    x=h.timestamp_s.to_numpy(float);y=h[feature].to_numpy(float)
    slope=float(np.polyfit(x-x[0],y,1)[0]);current=float(y[-1])
    if current>=threshold:return {'status':'threshold_reached','rul_hours':0.0,'slope_per_hour':slope}
    if slope<=0:return {'status':'no_detectable_degradation','rul_hours':None,'slope_per_hour':slope}
    return {'status':'exploratory_estimate','rul_hours':round((threshold-current)/slope,2),'slope_per_hour':slope}

def predict(df, models_path, output):
    m=joblib.load(models_path);windows=make_windows(df)
    X=windows[m['features']]
    windows['predicted_fault']=m['model'].predict(X)
    windows['model_max_class_probability']=m['model'].predict_proba(X).max(axis=1).round(3)
    windows['anomaly_flag']=(m['anomaly'].predict(X)==-1)
    windows['anomaly_score']=m['anomaly'].decision_function(X).round(4)
    output_rows=[]
    for _,w in windows.iterrows():
        row=w.to_dict();row.update(health_scores(w))
        output_rows.append(row)
    pd.DataFrame([{**{k:v for k,v in x.items() if k!='subsystems'},**{'health_'+k:v for k,v in x['subsystems'].items()}} for x in output_rows]).to_csv(output,index=False)
    return len(output_rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['generate','train','predict','rul','all'],default='all');p.add_argument('--input',default='synthetic.csv');p.add_argument('--model-dir',default='artifacts');p.add_argument('--output',default='predictions.csv');a=p.parse_args()
    if a.mode in ('generate','all'):
        make_data().to_csv(a.input,index=False);print('Generated',a.input)
    if a.mode in ('train','all'):
        print('Train',train(pd.read_csv(a.input),a.model_dir))
    if a.mode in ('predict','all'):
        print('Predicted windows',predict(pd.read_csv(a.input),Path(a.model_dir)/'models.joblib',a.output))
    if a.mode in ('rul','all'):
        demo=pd.DataFrame({'timestamp_s':np.arange(36,dtype=float),'vibration_rms':np.linspace(.6,2.4,36)})
        print('Synthetic hourly degradation RUL',rul_from_history(demo,threshold=3.0))
if __name__=='__main__':main()
