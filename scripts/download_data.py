"""Download a pinned UAV subset and NASA's original battery archive member.
No synthetic fallback. Raw files are local-only; sources and SHA256 are recorded.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NASA = 'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip'
REPO = 'tiiuae/UAV-Realistic-Fault-Dataset'

def fetch(url, headers=None):
    req = urllib.request.Request(url, headers={'User-Agent': 'PHM-research-prototype', **(headers or {})})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return response.read()
        except Exception:
            if attempt == 3: raise
            time.sleep(attempt + 1)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class RemoteZip(io.RawIOBase):
    def __init__(self, url, size=None, workers=8):
        self.url, self.pos, self.workers = url, 0, workers
        if size is not None:
            self.size = size
            return
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as r:
            self.size = int(r.headers['Content-Length'])
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        return self.pos
    def part(self, bounds):
        start, end = bounds
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end - 1}'})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    if r.status != 206: raise ValueError('Server does not support range downloads')
                    data = r.read()
                break
            except Exception:
                if attempt == 3: raise
                time.sleep(attempt + 1)
        if len(data) != end - start: raise ValueError('Truncated download')
        return data
    def read(self, n=-1):
        end = self.size if n < 0 else min(self.pos + n, self.size)
        if end <= self.pos: return b''
        blocks = [(i, min(i + 1024*1024, end)) for i in range(self.pos, end, 1024*1024)]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            data = b''.join(pool.map(self.part, blocks))
        self.pos = end
        return data

def battery():
    folder = ROOT / 'data/raw/nasa'; folder.mkdir(parents=True, exist_ok=True)
    archive = folder / 'BatteryAgingARC-FY08Q4.zip'
    member = '5. Battery Data Set/1. BatteryAgingARC-FY08Q4.zip'
    if not archive.exists():
        with zipfile.ZipFile(RemoteZip(NASA)) as z:
            payload = z.read(member)
        temp = archive.with_suffix('.part'); temp.write_bytes(payload); temp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            # Flatten only the four documented cells and the accompanying readme.
            base = Path(name).name
            if base in {'B0005.mat', 'B0006.mat', 'B0007.mat', 'B0018.mat'} or 'readme' in base.lower():
                (folder / base).write_bytes(z.read(name))
    meta = {'source': NASA, 'archive_member': member, 'sha256': digest(archive),
            'files': {p.name: digest(p) for p in sorted(folder.glob('B*.mat'))},
            'data_kind': 'real_lab', 'citation': 'B. Saha and K. Goebel (2007), NASA Battery Data Set',
            'license_note': 'Preserve NASA repository attribution; verify terms before redistribution.'}
    (folder / 'manifest.json').write_text(json.dumps(meta, indent=2))
    print('NASA samples ready:', folder, flush=True)

def uav(per_class=8):
    folder = ROOT / 'data/raw/tii'; folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        revision = old['revision']
    elif (folder / 'revision.txt').exists():
        revision = (folder / 'revision.txt').read_text().strip()
    else:
        revision = json.loads(fetch(f'https://api.github.com/repos/{REPO}/commits/main'))['sha']
    listing = json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{revision}?recursive=1'))
    listing['requested_revision'] = revision
    (folder / 'tree.json').write_text(json.dumps(listing))
    tree = listing['tree']
    blob_info = {x['path']: x for x in tree if x['type'] == 'blob'}
    (folder / 'revision.txt').write_text(revision)
    selected = []
    for label in range(5):
        paths = sorted(x['path'] for x in tree if x['path'].startswith(f'Dataset/{label}/') and x['path'].endswith('-SensorCombined.jsonl'))
        if len(paths) < per_class or per_class < 6: raise ValueError('Require at least six missions per class')
        indices = [round(i*(len(paths)-1)/(per_class-1)) for i in range(per_class)]
        selected.extend(paths[i] for i in indices)
    def download(path):
        dest = folder / path; dest.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://raw.githubusercontent.com/{REPO}/{revision}/{path}'
        if dest.exists():
            content = dest.read_bytes()
            actual = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
            if actual != blob_info[path]['sha']:
                raise ValueError(f'Existing file checksum mismatch; remove and redownload {dest}')
        else:
            content = RemoteZip(url, size=blob_info[path]['size'], workers=2).read()
            blob_hash = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
            if blob_hash != blob_info[path]['sha']: raise ValueError(f'Git blob checksum mismatch: {path}')
            temp = dest.with_suffix('.part'); temp.write_bytes(content); temp.replace(dest)
        print('UAV:', path.split('/')[2], flush=True)
        return {'path': path, 'url': url, 'sha256': digest(dest), 'class': int(path.split('/')[1])}
    with ThreadPoolExecutor(max_workers=5) as pool: files = list(pool.map(download, selected))
    manifest.write_text(json.dumps({'source': f'https://github.com/{REPO}', 'revision': revision,
        'data_kind': 'real_flight', 'files': files, 'selection': f'{per_class} chronologically spaced missions per class',
        'license_note': 'Repository has no explicit dataset license at inspection. Local research demo; obtain terms before redistributing raw data.'}, indent=2))
    print('UAV subset ready:', len(files), 'missions', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dataset', choices=['all','uav','battery'], default='all'); p.add_argument('--per-class', type=int, default=8)
    args = p.parse_args()
    if args.dataset in ('uav','all'): uav(args.per_class)
    if args.dataset in ('battery','all'): battery()
