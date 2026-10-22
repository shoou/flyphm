"""Download the pinned V13 vibrometer WAV subset, retaining attribution and hashes.
No entire 8 GB current archive, no synthetic fallback, and no mislabeled sensor fusion.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from phm.data import ROOT, read_json, write_json, sha256

DATASET='g28trvywnx'
VERSION=13
SELECTED_SPEEDS=[5,10,20,30,40,50,60]
PAGE=f'https://data.mendeley.com/datasets/{DATASET}/{VERSION}'
API=f'https://data.mendeley.com/public-api/datasets/{DATASET}'


def fetch(url):
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Referer':PAGE})
            with urllib.request.urlopen(req,timeout=60) as response:return response.read()
        except Exception:
            if attempt==3:raise
            time.sleep(attempt+1)


def save_file(file,path):
    expected=file['content_details']['sha256_hash']
    if path.exists() and sha256(path)==expected:return
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.part')
    for attempt in range(5):
        offset=temp.stat().st_size if temp.exists() else 0
        if offset==file['size'] and sha256(temp)==expected:
            temp.replace(path);return
        headers={'User-Agent':'Mozilla/5.0','Referer':PAGE}
        if offset:headers['Range']=f'bytes={offset}-'
        try:
            req=urllib.request.Request(file['content_details']['download_url'],headers=headers)
            with urllib.request.urlopen(req,timeout=30) as response:
                append=bool(offset and response.status==206)
                if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('Invalid resumed byte range')
                with temp.open('ab' if append else 'wb') as target:
                    while True:
                        chunk=response.read(65536)
                        if not chunk:break
                        target.write(chunk)
            if sha256(temp)!=expected:
                if temp.stat().st_size>=file['size']:temp.unlink()
                raise ValueError('SHA256 mismatch or truncated download: '+str(path))
            temp.replace(path);return
        except Exception:
            if attempt==4:raise
            time.sleep(attempt+1)


def download(root=None,workers=6):
    root=Path(root or ROOT/'data/raw/servo');root.mkdir(parents=True,exist_ok=True)
    catalog_path=root/'catalog_v13.json'
    if catalog_path.exists():catalog=read_json(catalog_path)
    else:
        catalog=json.loads(fetch(API+'?version=13'))
        if catalog.get('version')!=VERSION:raise ValueError('Endpoint returned a different dataset version; refusing unpinned data')
        write_json(catalog_path,catalog)
    if catalog.get('version')!=VERSION:raise ValueError('Wrong catalog version')
    files=catalog['files']
    for name in ['README.md','LICENSE.txt','metadata.csv','checksums_sha256.txt']:
        save_file(next(f for f in files if f['filename']==name),root/name)
        print('Metadata ready:',name,flush=True)
    checksum={}
    for line in (root/'checksums_sha256.txt').read_text().splitlines():
        if not line.strip():continue
        digest,path=line.split(maxsplit=1);checksum[path.strip().lstrip('*').replace('\\','/').removeprefix('./')]=digest
    by_hash={f['content_details']['sha256_hash']:f for f in files}
    metadata=pd.read_csv(root/'metadata.csv');wav=metadata[(metadata['format']=='wav') & metadata.speed_percent.isin(SELECTED_SPEEDS)]
    labels={'normal':0,'loose_foundation':1,'suboptimal_control':2,'suboptimal_control_rt':3}
    print('Operations:',wav.operation.unique().tolist(),flush=True)
    recordings=[];jobs=[]
    for row in wav.to_dict('records'):
        path=row['new_path']
        if Path(path).is_absolute() or '..' in Path(path).parts:raise ValueError('Unsafe archive path')
        digest=checksum[path]
        f=by_hash[digest]
        if f['filename']!=Path(path).name:raise ValueError('Metadata/file mismatch')
        if row['operation'] not in labels:raise ValueError('Unknown condition: '+row['operation'])
        recordings.append({'record_id':row['condition']+'_'+f"{int(row['speed_percent']):03d}",
            'condition':row['condition'],'label':labels[row['operation']],'speed_pct':int(row['speed_percent']),
            'reversal':row['reversal']=='yes','path':path,'sha256':digest,'file_id':f['id'],
            'sample_rate':int(row['sample_rate_hz']),'duration_s':float(row['duration_s'])})
        jobs.append((f,root/path))
    # Save the complete fixed selection before collecting any model scores.
    write_json(root/'download_plan.json',{'version':VERSION,'sensor':'vibrometer_wav','recordings':recordings,
        'test_speeds':[20,50],'validation_speeds':[10,40],'policy':'fixed seven common speed setpoints, balanced four-state/direction coverage; chosen before any scores', 'selected_speeds':SELECTED_SPEEDS})
    errors=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(save_file,*job):job[1] for job in jobs}
        for i,future in enumerate(as_completed(pending),1):
            try:future.result();print(f'{i}/{len(jobs)} ready: {pending[future].relative_to(root)}',flush=True)
            except Exception as exc:errors.append(str(pending[future])+': '+str(exc));print('FAILED:',errors[-1],flush=True)
    if errors:raise RuntimeError('Incomplete download; rerun resumes verified files.\n'+'\n'.join(errors))
    manifest={'dataset':DATASET,'version':VERSION,'doi':'10.17632/g28trvywnx.13','source':PAGE,'license':'CC BY 4.0',
        'citation':'Zhilevski, Marin; Slavov, Danail; Yordanov, Nikolay (2026), Multi-Sensor Condition-Monitoring Dataset of a Brushed DC Servo Motor, Mendeley Data, V13.',
        'sensor':'vibrometer_wav','selected_speeds':SELECTED_SPEEDS,'recordings':recordings,'catalog_sha256':sha256(catalog_path),'metadata_sha256':sha256(root/'metadata.csv'),
        'data_kind':'real_lab_no_load','synchronized':False,'independent_motors':1}
    write_json(root/'manifest.json',manifest)
    print('Servo dataset ready:',len(recordings),'recordings',flush=True)
    return root


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
    download(workers=args.workers)
