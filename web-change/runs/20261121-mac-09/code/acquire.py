"""Acquire exact upstream test pages and styles; never patch source content."""
from pathlib import Path
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import sys,json
RUN=Path(__file__).resolve().parents[1];PROJECT=RUN.parents[1];DATA=PROJECT/'data/contrast-source-03';OUT=RUN/'outputs'
sys.path.insert(0,str(RUN.parent/'20261121-mac-07/code'))
from acquire_pairs import get
from pair_audit import git_blob_sha

def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 cfg=json.loads((RUN/'protocol.json').read_text());specs=json.loads((RUN/'source_specs.json').read_text())
 for p,h in cfg['word_guard'].items():assert sha256(Path(p).read_bytes()).hexdigest()==h
 DATA.mkdir(exist_ok=True);(DATA/'blobs').mkdir(exist_ok=True);jobs={}
 for case in specs['cases']:
  for version in case['versions']:
   for f in version['files']:
    h=f['blob_sha'];jobs.setdefault(h,(f"https://raw.githubusercontent.com/{case['repository']}/{version['commit']}/{f['path']}",h,DATA/'blobs'/h))
 with ThreadPoolExecutor(max_workers=4) as pool:downloaded=list(pool.map(lambda x:get(*x),jobs.values()))
 records=[]
 for case in specs['cases']:
  for v in case['versions']:
   for f in v['files']:
    dest=DATA/case['id']/v['name']/f['path'];dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.hardlink_to(DATA/'blobs'/f['blob_sha'])
    assert git_blob_sha(dest.read_bytes())==f['blob_sha'];records.append({'case':case['id'],'version':v['name'],'path':f['path'],'git_blob_sha':f['blob_sha'],'sha256':sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size})
 dump(OUT/'source_files.json',records);dump(OUT/'downloads.json',downloaded)
 dump(RUN/'source_lock.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'hashes':{str(p.relative_to(PROJECT)):sha256(p.read_bytes()).hexdigest() for p in [RUN/'protocol.json',RUN/'source_specs.json',RUN/'code/acquire.py',RUN.parent/'20261121-mac-07/code/acquire_pairs.py',RUN.parent/'20261121-mac-07/code/pair_audit.py']}})
 print(json.dumps({'cases':len(specs['cases']),'source_file_references':len(records),'unique_blobs':len(downloaded),'source_bytes_edited':0}),flush=True)
if __name__=='__main__':main()
