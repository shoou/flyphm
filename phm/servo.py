"""Servo vibration diagnosis, pinned real recordings and speed-held-out evaluation.
No synchronized control residual, physical vibration calibration, or RUL is inferred.
"""
from pathlib import Path
import shutil
import time
import joblib
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from phm.data import ROOT, read_json, write_json, sha256
from phm.training import new_run, classification

RAW=ROOT/'data/raw/servo'
PROCESSED=ROOT/'data/processed/servo'
LABELS={0:'正常运行',1:'底座松动',2:'速度调节不佳',3:'电流调节不佳'}
# Fixed before any feature extraction or model score is inspected. All states/directions
# at the same speed stay together. A recording is not an independent physical motor.
TEST_SPEEDS=[20,50]
VALIDATION_SPEEDS=[10,40]
FEATURES=['rms','std','peak','crest','kurtosis','skew','zero_crossing','dominant_hz','spectral_entropy',
          'band_0_100','band_100_500','band_500_2000','band_2000_10000','band_10000_nyquist']
LIMITATIONS=[
    '同一无负载台架、每工况每传感器仅一次录制；按转速隔离，不代表跨设备或重复试验验证。',
    '仅使用振动仪 WAV 通道；不同传感器顺序采集，未做同步融合。',
    '幅值为数字归一化幅值，非校准加速度或速度；分类概率未校准。',
    '无同步指令/反馈轨迹、故障发生时刻和全寿命退化标签，不计算跟踪残差、检测延迟或 RUL。',
]


def split_for_speed(speed):
    return 'test' if speed in TEST_SPEEDS else 'validation' if speed in VALIDATION_SPEEDS else 'train'


def waveform(path):
    rate,x=wavfile.read(path)
    if not 20000<=rate<=192000:raise ValueError('不支持的 WAV 采样率')
    if x.ndim==2:x=x[:,0]  # fixed channel, never choose channel using fault label
    if x.ndim!=1:raise ValueError('WAV 通道维度无效')
    if np.issubdtype(x.dtype,np.integer):
        if x.dtype==np.uint8:x=(x.astype(float)-128)/128
        else:x=x.astype(float)/max(abs(np.iinfo(x.dtype).min),np.iinfo(x.dtype).max)
    else:x=x.astype(float)
    if not np.isfinite(x).all():raise ValueError('WAV 存在非有限值')
    return rate,x


def signal_features(x,rate):
    if len(x)!=rate or not np.isfinite(x).all():raise ValueError('必须提供完整的 1 秒有限值波形')
    x=x-np.mean(x);std=float(np.std(x));rms=float(np.sqrt(np.mean(x*x)));peak=float(np.max(np.abs(x)))
    if rms<1e-10:raise ValueError('静音或零能量窗口，无法评估')
    f,power=welch(x,fs=rate,nperseg=min(4096,len(x)))
    total=float(power.sum())
    if total<=0:raise ValueError('无有效频谱能量')
    norm=power/total
    result={'rms':rms,'std':std,'peak':peak,'crest':peak/rms,'kurtosis':float(kurtosis(x,fisher=False)),
            'skew':float(skew(x)),'zero_crossing':float(np.mean(np.signbit(x[1:])!=np.signbit(x[:-1]))),
            'dominant_hz':float(f[np.argmax(power)]),'spectral_entropy':float(-(norm*np.log(norm+1e-15)).sum()/np.log(len(norm)))}
    for name,low,high in [('band_0_100',0,100),('band_100_500',100,500),('band_500_2000',500,2000),('band_2000_10000',2000,10000),('band_10000_nyquist',10000,rate/2+1)]:
        result[name]=float(power[(f>=low)&(f<high)].sum()/total)
    if not np.isfinite(list(result.values())).all():raise ValueError('非有限特征')
    return result


def prepare(raw=None,processed=None):
    raw=Path(raw or RAW);processed=Path(processed or PROCESSED)
    manifest=read_json(raw/'manifest.json')
    if manifest['version']!=13 or manifest['sensor']!='vibrometer_wav':raise ValueError('数据版本或模态不匹配')
    ids=[r['record_id'] for r in manifest['recordings']];hashes=[r['sha256'] for r in manifest['recordings']]
    if len(set(ids))!=len(ids) or len(set(hashes))!=len(hashes):raise ValueError('重复记录或波形内容，不允许跨工况重复样本')
    windows=[];quality=[];traces=[]
    for rec in manifest['recordings']:
        path=raw/rec['path']
        if path.resolve().is_relative_to(raw.resolve()) is False:raise ValueError('无效数据路径')
        if sha256(path)!=rec['sha256']:raise ValueError('原始波形校验失败：'+rec['record_id'])
        rate,x=waveform(path);count=len(x)//rate;usable=0;reasons=[]
        if 'sample_rate' in rec and rate!=rec['sample_rate']:raise ValueError('采样率与发布元数据不符')
        if 'duration_s' in rec and abs(len(x)/rate-rec['duration_s'])>.001:raise ValueError('录音时长与发布元数据不符')
        if count<3:raise ValueError('记录不足 3 秒')
        info={k:rec[k] for k in ['record_id','condition','label','speed_pct','reversal']}
        info['split']=split_for_speed(rec['speed_pct'])
        for i in range(1,count):  # exclude documented quiet startup, fixed before scoring
            try:
                chunk=x[i*rate:(i+1)*rate]
                if (np.abs(chunk)>=.999).mean()>.01:raise ValueError('数字幅值削顶超过 1%，窗口无法评估')
                features=signal_features(chunk,rate)
            except ValueError as exc:reasons.append({'second':i+1,'reason':str(exc)});continue
            windows.append({**info,'timestamp_s':float(i+1),**features});usable+=1
        # Min/max per 10 ms preserves vibration envelope without exporting huge raw WAV.
        hop=max(1,rate//100)
        for i in range(0,len(x)-hop+1,hop):
            chunk=x[i:i+hop];traces.append({'record_id':rec['record_id'],'timestamp_s':i/rate,'minimum':float(chunk.min()),'maximum':float(chunk.max())})
        quality.append({**info,'sample_rate':rate,'duration_s':len(x)/rate,'windows':count-1,'trimmed_start_s':1,'usable_windows':usable,
                        'discarded_tail_s':(len(x)%rate)/rate,'clipped_fraction':float((np.abs(x)>=.999).mean()),'unknown_windows':reasons})
    w=pd.DataFrame(windows)
    if w.empty:raise ValueError('没有可评估波形')
    if w.groupby('record_id').split.nunique().max()!=1:raise ValueError('记录跨集合')
    for role in ['train','validation','test']:
        if set(w[w.split==role].label)!=set(LABELS):raise ValueError('集合缺少类别：'+role)
    processed.mkdir(parents=True,exist_ok=True)
    w.to_parquet(processed/'windows.parquet',index=False)
    pd.DataFrame(traces).to_parquet(processed/'waveforms.parquet',index=False)
    write_json(processed/'quality.json',{'records':quality,'sensor':'vibrometer_wav','channel':0,'window_seconds':1,'stride_seconds':1,'trimmed_start_s':1,'max_window_clipping_fraction':.01,'limitations':LIMITATIONS})
    shutil.copy2(raw/'manifest.json',processed/'manifest.json')
    return processed


def record_predictions(w):
    rows=[]
    for rid,g in w.groupby('record_id',sort=True):
        counts=g.predicted_label.value_counts();prediction=int(counts[counts==counts.max()].index.min())
        rows.append({'record_id':rid,'label':int(g.label.iloc[0]),'predicted_label':prediction,'split':g.split.iloc[0],
                     'speed_pct':int(g.speed_pct.iloc[0]),'reversal':bool(g.reversal.iloc[0]),'windows':len(g),
                     'vote_fraction':float(counts.max()/len(g))})
    return pd.DataFrame(rows)


def train(processed=None,run_root=None,trees=160):
    processed=Path(processed or PROCESSED);w=pd.read_parquet(processed/'windows.parquet')
    # Verify supplied split rather than trusting mutable parquet metadata.
    if not w.split.equals(w.speed_pct.map(split_for_speed)):raise ValueError('划分不符合冻结转速方案')
    if w.groupby('record_id').split.nunique().max()!=1:raise ValueError('同一记录跨集合泄漏')
    for role in ['train','validation','test']:
        if set(w[w.split==role].label)!=set(LABELS):raise ValueError('训练/验证/测试须覆盖全部类别')
    config={'trees':trees,'seed':42,'test_speeds':TEST_SPEEDS,'validation_speeds':VALIDATION_SPEEDS,
            'data_sha256':sha256(processed/'windows.parquet'),'servo_code_sha256':sha256(Path(__file__)),
            'features':FEATURES,'selection':'validation recording-level macro F1; tie uses first candidate; test never selects model'}
    run=new_run('servo',config,run_root);start=time.monotonic()
    try:
        tr=w[w.split=='train'];va=w[w.split=='validation']
        candidates=[('random_forest',RandomForestClassifier(n_estimators=trees,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=2)),
                    ('logistic_regression',make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42)))]
        selection=[];trained={}
        for name,model in candidates:
            model.fit(tr[FEATURES],tr.label);v=va.copy();v['predicted_label']=model.predict(v[FEATURES]);r=record_predictions(v)
            selection.append({'model':name,'validation_record_macro_f1':classification(r.label,r.predicted_label,list(LABELS))['macro_f1']})
            trained[name]=model
        chosen=max(selection,key=lambda x:x['validation_record_macro_f1'])['model'];model=trained[chosen]
        baseline=DummyClassifier(strategy='most_frequent').fit(tr[FEATURES],tr.label)
        w['predicted_label']=model.predict(w[FEATURES]);w['class_probability']=model.predict_proba(w[FEATURES]).max(axis=1)
        w['baseline_label']=baseline.predict(w[FEATURES]);rp=record_predictions(w)
        results={}
        for role in ['train','validation','test']:
            g=w[w.split==role];r=rp[rp.split==role]
            results[role]={'windows':classification(g.label,g.predicted_label,list(LABELS)),
                           'records':classification(r.label,r.predicted_label,list(LABELS)),
                           'baseline_windows':classification(g.label,g.baseline_label,list(LABELS)),
                           'n_records':len(r),'n_windows':len(g)}
        te=rp[rp.split=='test'];per_speed=[]
        for speed,g in te.groupby('speed_pct'):
            per_speed.append({'speed_pct':int(speed),**classification(g.label,g.predicted_label,list(LABELS))})
        importance=model.feature_importances_ if hasattr(model,'feature_importances_') else np.abs(model[-1].coef_).mean(axis=0)
        pd.DataFrame({'feature':FEATURES,'importance':importance}).sort_values('importance',ascending=False).to_csv(run/'feature_importance.csv',index=False)
        joblib.dump({'model':model,'features':FEATURES,'labels':LABELS,'sensor':'vibrometer_wav','channel':0},run/'models.joblib')
        w.to_parquet(run/'predictions.parquet',index=False);rp.to_csv(run/'record_predictions.csv',index=False)
        for src,dst in [('waveforms.parquet','waveforms.parquet'),('quality.json','data_quality.json'),('manifest.json','data_manifest.json')]:shutil.copy2(processed/src,run/dst)
        metrics={'task':'servo','dataset':'DC Servo Motor · Mendeley V13 · 振动仪 WAV','selected_model':chosen,'selection':selection,'results':results,'per_test_speed':per_speed,
                 'recordings':len(rp),'windows':len(w),'labels':LABELS,'seconds':time.monotonic()-start,'limitations':LIMITATIONS,
                 'source':'https://data.mendeley.com/datasets/g28trvywnx/13','license':'CC BY 4.0','split_unit':'speed_pct across all conditions/directions',
                 'test_speeds':TEST_SPEEDS,'validation_speeds':VALIDATION_SPEEDS,'selected_speeds':sorted(int(x) for x in w.speed_pct.unique()),'rul':None,'detection_delay':None}
        write_json(run/'metrics.json',metrics);write_report(run,metrics,rp)
        write_json(run/'status.json',{'status':'complete'})
        return run
    except Exception as exc:
        write_json(run/'status.json',{'status':'failed','error':str(exc)});raise


def score_case(run,source,case,cutoff=0):
    bundle=joblib.load(Path(run)/'models.joblib');w=pd.read_parquet(Path(source)/'predictions.parquet')
    w=w[w.record_id==case].copy()
    if cutoff:w=w[w.timestamp_s<=cutoff]
    if w.empty:raise ValueError('截止时间内无有效伺服窗口')
    if not np.isfinite(w[bundle['features']]).all().all():raise ValueError('伺服特征存在缺失或非有限值')
    w['predicted_label']=bundle['model'].predict(w[bundle['features']]);w['class_probability']=bundle['model'].predict_proba(w[bundle['features']]).max(axis=1)
    return w.sort_values('timestamp_s')




def fault_events(w):
    """Causal review rule: three consecutive same-fault 1 s windows, no gap bridging."""
    events=[];label=0;count=0;previous=None;current=None
    for row in w.sort_values('timestamp_s').itertuples():
        t=float(row.timestamp_s);pred=int(row.predicted_label)
        if pred!=label or previous is None or t-previous>1.5:
            label=pred;count=0;current=None
        count+=1
        if label and count==3:
            current={'key':f'servo_{label}_{t-2:g}','title':'伺服振动诊断建议复核：'+LABELS[label],'priority':'中',
                     'evidence':{'predicted_label':label,'condition':LABELS[label],'start_s':t-2,'confirmed_s':t,'end_s':t,
                                 'rule':'连续 3 个相同异常分类窗口；仅人工复核规则','probability':float(row.class_probability)}}
            events.append(current)
        elif current is not None:current['evidence']['end_s']=t
        previous=t
    return events


def analyze_binding(binding,folder,run,source):
    w=score_case(run,source,binding['case_id'],binding['cutoff'])
    w.to_parquet(folder/'trend.parquet',index=False)
    events=fault_events(w);latest=w.iloc[-1]
    reference=int(w.label.iloc[0])
    for event in events:
        event['evidence']['reference_class']=LABELS[reference]
        event['evidence']['reference_note']='公开记录标签，仅事后对照，未参与模型输入'
    manifest=read_json(Path(source)/'data_manifest.json')
    raw=next(r for r in manifest['recordings'] if r['record_id']==binding['case_id'])
    result={'family':'servo','status':'需复核' if events else '已评估','score':None,'score_label':'无标定健康分','rul':None,
        'reference_class':reference,'reference_class_name':LABELS[reference],
        'latest_class':int(latest.predicted_label),'class_name':LABELS[int(latest.predicted_label)],'probability':float(latest.class_probability),
        'windows':len(w),'fault_fraction':float((w.predicted_label!=0).mean()),'alert_events':len(events),
        'recommended_action':'人工核查固定螺栓、驱动调节参数和振动；由专业人员确认，模型不自动调整控制器。' if events else '本段未触发连续异常复核规则，继续监测。',
        'evidence':'单台架振动状态分类；无标定 HI、指令跟踪残差和寿命结论',
        'provenance':{'model_run':Path(run).name,'source_kind':'servo_case','source_ref':Path(source).name,'case_id':binding['case_id'],
            'cutoff':binding['cutoff'],'input_sha256':raw['sha256'],'features_sha256':sha256(Path(source)/'predictions.parquet'),
            'model_sha256':sha256(Path(run)/'models.joblib'),'source':manifest['source'],'speed_pct':raw['speed_pct'],'reversal':raw['reversal'],
            'sensor':'vibrometer_wav','split':str(latest.split),'labels_used_for_inference':False}}
    quality=read_json(Path(source)/'data_quality.json')
    q=next(r for r in quality['records'] if r['record_id']==binding['case_id'])
    unknown=[r for r in q['unknown_windows'] if not binding['cutoff'] or r['second']<=binding['cutoff']]
    result['provenance']['unknown_windows']=len(unknown)
    if unknown:
        result['quality']='部分窗口无法评估'
        if not events:result['status']='部分评估'
    return result,events


def write_report(run,metrics,records):
    from html import escape
    summary=[]
    for role,r in metrics['results'].items():
        summary.append({'集合':role,'记录':r['n_records'],'窗口':r['n_windows'],'记录 Macro-F1':r['records']['macro_f1'],'窗口 Macro-F1':r['windows']['macro_f1'],'多数类基线 F1':r['baseline_windows']['macro_f1']})
    matrix=pd.DataFrame(metrics['results']['test']['records']['confusion_matrix'],index=LABELS.values(),columns=LABELS.values())
    body='<h1>伺服执行机构诊断 · 验证报告</h1><p>实验 '+escape(Path(run).name)+'</p>'
    body+='<p>测试转速 20%、50%；验证转速 10%、40%；其余训练。单台无负载实验设备，非跨设备验证。仅验证集选择模型。</p>'
    body+='<p>模型：'+escape(metrics['selected_model'])+'</p>'+pd.DataFrame(summary).to_html(index=False,float_format=lambda x:f'{x:.4f}')
    body+='<h2>测试记录混淆矩阵</h2><p>行：真实状态；列：预测状态。</p>'+matrix.to_html()
    body+='<h2>逐记录结果（含误判）</h2>'+records[records.split=='test'].to_html(index=False)
    body+='<h2>适用边界</h2><ul>'+''.join('<li>'+escape(x)+'</li>' for x in metrics['limitations'])+'</ul>'
    body+='<p>数据：Zhilevski, Slavov, Yordanov (2026), Multi-Sensor Condition-Monitoring Dataset of a Brushed DC Servo Motor, V13, DOI 10.17632/g28trvywnx.13, CC BY 4.0.</p>'
    body+='<a href="https://data.mendeley.com/datasets/g28trvywnx/13">发布方数据页面</a>'
    (Path(run)/'validation_report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>伺服诊断验证报告</title><style>body{font:15px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#17324b}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border:1px solid #dce5ef;text-align:left}th{background:#eaf2f7}h2{margin-top:32px}</style>'+body+'</html>')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:print('Prepared',prepare(),flush=True)
    print('Completed',train(),flush=True)

