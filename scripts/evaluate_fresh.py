"""Freeze a derived policy, then fetch previously unused missions for one holdout evaluation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
import pandas as pd
from scripts.download_data import fetch, RemoteZip, REPO
from phm.data import ROOT, read_json, write_json, sha256, prepare_uav
from phm.alerts import score_conditioned, confirm_alerts, alarm_metrics
from phm.training import classification
from phm.experiments import compare_runs, comparison_html


def evaluate(run):
    run=Path(run); policy_path=run/'alert_policy.joblib'
    policy=joblib.load(policy_path); baseline=joblib.load(run/'models.joblib')
    if (run/'fresh_evaluation.json').exists():
        print('Existing sealed evaluation:',run/'fresh_evaluation.json'); return
    raw=ROOT/'data/raw'/('fresh_'+run.name); raw.mkdir(parents=True,exist_ok=True)
    manifest=read_json(ROOT/'data/raw/tii/manifest.json'); revision=manifest['revision']
    used=set(read_json(run/'split.json'))
    # A cohort previously evaluated by another experiment must not become "fresh" again.
    for prior in (ROOT/'runs').glob('*/fresh_evaluation.json'):
        used.update(read_json(prior).get('mission_ids',[]))
    plan_path=raw/'evaluation_plan.json'
    if plan_path.exists():
        plan=read_json(plan_path)
        if plan['policy_sha256']!=sha256(policy_path): raise ValueError('Policy changed after cohort selection')
    else:
        cache=ROOT/'data/raw/tii/tree.json'
        listing=read_json(cache) if cache.exists() else json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{revision}?recursive=1'))
        if cache.exists() and listing.get('requested_revision',listing['sha']) != revision: raise ValueError('Cached repository revision mismatch')
        if listing.get('truncated'): raise ValueError('Truncated repository listing')
        tree=listing['tree']
        selected=[]
        for label in range(5):
            choices=sorted([x for x in tree if x['path'].startswith(f'Dataset/{label}/') and x['path'].endswith('-SensorCombined.jsonl') and Path(x['path']).parent.name not in used],key=lambda x:x['path'])
            if len(choices)<2: raise ValueError('Insufficient unused missions')
            selected+=choices[-2:]
        plan={'policy_sha256':sha256(policy_path),'model_sha256':sha256(run/'models.joblib'),'revision':revision,
              'selection':'Two chronologically latest unused missions per class, selected before reading signals or evaluation.', 'files':selected}
        write_json(plan_path,plan)
    if plan['model_sha256']!=sha256(run/'models.joblib'): raise ValueError('Classifier changed after cohort selection')
    def download(entry):
        path=entry['path']; dest=raw/path; dest.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://raw.githubusercontent.com/{REPO}/{revision}/{path}'
        if not dest.exists():
            content=RemoteZip(url,size=entry['size'],workers=2).read()
            if hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()!=entry['sha']: raise ValueError('Git checksum mismatch')
            temp=dest.with_suffix('.part');temp.write_bytes(content);temp.replace(dest)
        content=dest.read_bytes()
        if hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()!=entry['sha']: raise ValueError('Git checksum mismatch')
        print('Fresh mission:',Path(path).parent.name,flush=True)
        return {'path':path,'sha256':sha256(dest),'class':int(path.split('/')[1]),'url':url}
    with ThreadPoolExecutor(max_workers=5) as pool: files=list(pool.map(download,plan['files']))
    write_json(raw/'manifest.json',{'source':manifest['source'],'revision':revision,'files':files})
    processed=prepare_uav(raw,ROOT/'data/processed'/('fresh_'+run.name))
    w=pd.read_parquet(processed/'windows.parquet')
    if set(w.mission_id)&set(read_json(run/'split.json')): raise ValueError('Holdout overlaps fitting/selection missions')
    w['predicted_label']=baseline['classifier'].predict(w[baseline['features']])
    w['class_probability']=baseline['classifier'].predict_proba(w[baseline['features']]).max(axis=1)
    w['anomaly_score']=-baseline['anomaly'].score_samples(w[baseline['features']])
    w['anomaly_threshold']=baseline['anomaly_threshold'];w['anomaly_flag']=w.anomaly_score>w.anomaly_threshold
    w['raw_anomaly_flag']=w.anomaly_flag
    if policy['conditioned'] is not None: candidate=score_conditioned(w,policy['conditioned'])
    else: candidate=w.copy()
    candidate,events=confirm_alerts(candidate,policy['confirm'],policy['release'],policy['merge_gap_s'])
    report={'status':'frozen_policy_new_missions','policy_sha256':sha256(policy_path), 'selection':plan['selection'],
            'mission_ids':sorted(w.mission_id.unique()),'missions':int(w.mission_id.nunique()),'windows':len(w),
            'classification':classification(w.label,w.predicted_label,list(range(5))),
            'baseline':alarm_metrics(w,'anomaly_flag'),'candidate':alarm_metrics(candidate,events=events),
            'limitations':'Previously unused missions from the same public platform; not independent aircraft or verified onset labels. Do not retune on these scores.'}
    candidate.to_parquet(run/'fresh_predictions.parquet',index=False);events.to_parquet(run/'fresh_events.parquet',index=False)
    write_json(run/'fresh_evaluation.json',report)
    comparison=compare_runs(ROOT/'runs'/policy['parent_run'],run)
    write_json(run/'comparison.json',comparison)
    (run/'comparison.html').write_text(comparison_html(comparison),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    evaluate(parser.parse_args().run)
