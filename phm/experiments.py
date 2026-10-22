"""Derived alarm experiments and comparisons against an immutable parent split."""
from pathlib import Path
import copy
import html
import itertools
import json
import shutil
import time
import joblib
import pandas as pd
from phm.data import ROOT, read_json, write_json, sha256
from phm.training import new_run, list_runs
from phm.alerts import fit_conditioned, score_conditioned, confirm_alerts, alarm_metrics


def improve_alerts(parent, run_root=None):
    parent=Path(parent)
    if read_json(parent/'status.json')['status']!='complete': raise ValueError('Parent run must be complete')
    if (parent/'alert_policy.joblib').exists(): raise ValueError('Choose an original baseline, not another derived alarm run')
    predictions=pd.read_parquet(parent/'predictions.parquet')
    base=joblib.load(parent/'models.joblib')
    parent_config=read_json(parent/'config.json')
    config={**parent_config,'parent_run':parent.name,'parent_model_sha256':sha256(parent/'models.joblib'),
            'fixed_split_sha256':sha256(parent/'split.json'),'selection_set':'validation only',
            'evaluation_status':'previously_seen_test_regression',
            'selection_rule':'Minimize healthy alarm fraction then events, subject to mission recall >= baseline and fault window recall >= baseline - 0.05; baseline fallback.',
            'regime_note':'Low/high angular-motion proxy; not annotated flight phases.'}
    run=new_run('uav',config,run_root); started=time.monotonic()
    try:
        train=predictions[predictions.split=='train']; val=predictions[predictions.split=='validation']
        raw=val.copy(); raw['raw_anomaly_flag']=raw.anomaly_flag; raw['anomaly_threshold']=base['anomaly_threshold']
        raw,raw_events=confirm_alerts(raw,1,1,0)
        reference=alarm_metrics(raw,events=raw_events)
        rows=[]; best=None; candidates={}
        for quantile in [.95,.975,.99]:
            bundle=fit_conditioned(train,val,base['features'],quantile)
            scored=score_conditioned(val,bundle)
            for confirm,release in itertools.product([2,3,5],[2,3]):
                evaluated,events=confirm_alerts(scored,confirm,release,3)
                metrics=alarm_metrics(evaluated,events=events)
                eligible=(metrics['fault_mission_recall']>=reference['fault_mission_recall'] and
                          metrics['fault_window_recall']>=reference['fault_window_recall']-.05)
                key=f'q{quantile}_c{confirm}_r{release}'
                rows.append({'candidate':key,'quantile':quantile,'confirm':confirm,'release':release,'eligible':eligible,**metrics})
                candidates[key]={'conditioned':bundle,'confirm':confirm,'release':release,'merge_gap_s':3.}
        rows.append({'candidate':'baseline','quantile':.95,'confirm':1,'release':1,'eligible':True,**reference})
        eligible=[r for r in rows if r['eligible']]
        chosen=min(eligible,key=lambda r:(r['healthy_alarm_fraction'],r['false_alarm_events'],-r['fault_window_recall'],r['confirm']))
        policy=candidates.get(chosen['candidate'],{'conditioned':None,'confirm':1,'release':1,'merge_gap_s':0.})
        policy['parent_run']=parent.name; policy['features']=base['features']; policy['selection_candidate']=chosen['candidate']
        if policy['conditioned'] is None:
            scored=predictions.copy(); scored['raw_anomaly_flag']=scored.anomaly_flag
            scored['anomaly_threshold']=base['anomaly_threshold']; scored['operating_regime']='global_fallback'
        else: scored=score_conditioned(predictions,policy['conditioned'])
        evaluated,events=confirm_alerts(scored,policy['confirm'],policy['release'],policy['merge_gap_s'])
        evaluated.to_parquet(run/'predictions.parquet',index=False); events.to_parquet(run/'alarm_events.parquet',index=False)
        joblib.dump(policy,run/'alert_policy.joblib')
        for name in ['models.joblib','split.json','telemetry.parquet','feature_importance.csv','per_mission.csv','data_quality.json']:
            if (parent/name).exists(): shutil.copy2(parent/name,run/name)
        pd.DataFrame(rows).to_csv(run/'validation_candidates.csv',index=False)
        m=copy.deepcopy(read_json(parent/'metrics.json')); m.update(run_id=run.name,duration_s=time.monotonic()-started)
        m['evaluation_status']='previously_seen_test_regression'
        m['alert_improvement']={'parent_run':parent.name,'selected_candidate':chosen['candidate'],
             'confirm_windows':policy['confirm'],'release_windows':policy['release'],'merge_gap_s':policy['merge_gap_s'],
             'selection_set':'validation','regime_note':config['regime_note'],'by_split':{}}
        for role in ['validation','test']:
            current=evaluated[evaluated.split==role]
            these=events[events.mission_id.isin(current.mission_id)]
            m['alert_improvement']['by_split'][role]={
                'baseline':alarm_metrics(current,'anomaly_flag'), 'candidate':alarm_metrics(current,events=these)}
        m['limitations']+=['Alarm thresholds and confirmation parameters selected on validation only. Existing test is a regression comparison, not a new untouched holdout.',
                            'Operating regimes use current gyro magnitude, not verified takeoff/landing phase labels.']
        write_json(run/'metrics.json',m); write_json(run/'status.json',{'status':'complete'})
        comparison=compare_runs(parent,run)
        write_json(run/'comparison.json',comparison)
        (run/'comparison.html').write_text(comparison_html(comparison),encoding='utf-8')
        return run
    except Exception as exc:
        write_json(run/'status.json',{'status':'failed','error':str(exc)}); raise


def compare_runs(left,right):
    left,right=Path(left),Path(right)
    lm,rm=read_json(left/'metrics.json'),read_json(right/'metrics.json')
    if lm['task']!='uav' or rm['task']!='uav': raise ValueError('This comparison currently supports UAV experiments')
    lc,rc=read_json(left/'config.json'),read_json(right/'config.json')
    if lc.get('data_sha256')!=rc.get('data_sha256'): raise ValueError('Dataset fingerprints differ; metrics cannot be compared as a controlled experiment')
    if read_json(left/'split.json')!=read_json(right/'split.json'): raise ValueError('Mission splits differ; comparison refused')
    rows=[]
    lp,rp=pd.read_parquet(left/'predictions.parquet'),pd.read_parquet(right/'predictions.parquet')
    keys=['mission_id','timestamp_s','label','split']
    if not lp[keys].sort_values(keys).reset_index(drop=True).equals(rp[keys].sort_values(keys).reset_index(drop=True)):
        raise ValueError('Evaluation windows or labels differ')
    for role in ['validation','test']:
        for run,m,p in [(left,lm,lp),(right,rm,rp)]:
            selected=p[p.split==role]
            if (run/'alarm_events.parquet').exists():
                events=pd.read_parquet(run/'alarm_events.parquet')
                events=events[events.mission_id.isin(selected.mission_id)]
                alarm=alarm_metrics(selected,events=events)
            else: alarm=alarm_metrics(selected,'anomaly_flag')
            rows.append({'run':run.name,'split':role,'macro_f1':m['results'][role]['macro_f1'],**alarm})
    return {'baseline':left.name,'candidate':right.name,'comparable':True,'rows':rows,
            'data_sha256':lc['data_sha256'],'split_sha256':sha256(left/'split.json'),
            'interpretation':'Validation chooses parameters. Previously viewed test is regression evidence only. Report recall and alarm duration alongside event count. No fault onset labels => no detection latency.',
            'fresh_evaluation':read_json(right/'fresh_evaluation.json') if (right/'fresh_evaluation.json').exists() else None}


def comparison_html(result):
    table=pd.DataFrame(result['rows']).to_html(index=False,escape=True,float_format=lambda v:f'{v:.4f}')
    return '<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>PHM 告警改进报告</title><style>body{font:16px system-ui;margin:36px;color:#172a45}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd6e1;padding:8px}pre{white-space:pre-wrap}</style><h1>PHM 告警版本对比</h1><p>固定数据与任务划分；验证集选参，已有测试集仅作回归对比。不能仅凭事件数下降宣称效果改善。</p>'+table+'<h2>完整配置与新任务评价</h2><pre>'+html.escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre></html>'

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--parent',required=True)
    print(improve_alerts(parser.parse_args().parent))
