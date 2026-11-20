"""Pinned actual-version sources only; reuse existing verified downloads without duplicating blobs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from datetime import datetime,timezone
import json,sys,shutil
import numpy as np
from PIL import Image
RUN=Path(__file__).resolve().parents[1];PROJECT=RUN.parents[1];DATA=PROJECT/'data/real-version-replay-02';OUT=RUN/'outputs'
sys.path.insert(0,str(RUN.parent/'20261121-mac-07/code'))
from acquire_pairs import get
from pair_audit import git_blob_sha,pixel_audit

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 specs=json.loads((RUN/'source_specs.json').read_text());cfg=json.loads((RUN/'protocol.json').read_text())
 for p,h in cfg['word_guard'].items():assert sha256(Path(p).read_bytes()).hexdigest()==h
 DATA.mkdir(exist_ok=True);(DATA/'blobs').mkdir(exist_ok=True);(DATA/'licenses').mkdir(exist_ok=True)
 dump(RUN/'source_lock.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'hashes':{str(p.relative_to(PROJECT)):sha256(p.read_bytes()).hexdigest() for p in [RUN/'protocol.json',RUN/'source_specs.json',RUN/'code/acquire.py',RUN.parent/'20261121-mac-07/code/acquire_pairs.py',RUN.parent/'20261121-mac-07/code/pair_audit.py']}})
 jobs={}
 for s in specs['png_pairs']:
  for rev,key in [(s['parent_commit'],'before_blob_sha'),(s['commit'],'after_blob_sha')]:
   h=s[key];dest=DATA/'blobs'/(h+'.png');old=PROJECT/'data/real-version-pairs-01/blobs'/(h+'.png')
   if old.exists() and not dest.exists():assert git_blob_sha(old.read_bytes())==h;dest.hardlink_to(old)
   jobs.setdefault(h,(f"https://raw.githubusercontent.com/{s['repository']}/{rev}/{s['path']}",h,dest))
 with ThreadPoolExecutor(max_workers=4) as pool:downloads=list(pool.map(lambda args:get(*args),jobs.values()))
 licenses=[]
 for s in [specs['ionic_license'],{'repository':specs['mdn']['repository'],**specs['mdn']['license']}]:
  dest=DATA/'licenses'/(s['repository'].replace('/','__')+'.txt');licenses.append({**s,**get(f"https://raw.githubusercontent.com/{s['repository']}/{s['commit']}/{s['path']}",s['blob_sha'],dest)})
 mdn=specs['mdn'];assetbank={}
 for s in mdn['assets']:
  dest=DATA/'blobs'/s['blob_sha'];get(f"https://raw.githubusercontent.com/{mdn['repository']}/{s['commit']}/{s['path']}",s['blob_sha'],dest);assetbank[s['path']]=dest
 source_records=[]
 for s in mdn['html']:
  folder=DATA/'mdn'/s['version'];folder.mkdir(parents=True,exist_ok=True);dest=folder/Path(s['path']).name;info=get(f"https://raw.githubusercontent.com/{mdn['repository']}/{s['commit']}/{s['path']}",s['blob_sha'],dest);source_records.append({**s,**info,'local_path':str(dest.relative_to(PROJECT))})
  for name,asset in assetbank.items():
   target=folder/Path(name).name
   if not target.exists():target.hardlink_to(asset)
 rows=[]
 for s in specs['png_pairs']:
  paths=[DATA/'blobs'/(s[k]+'.png') for k in ['before_blob_sha','after_blob_sha']];a,b=[np.asarray(Image.open(p).convert('RGB')) for p in paths];rows.append({**s,**pixel_audit(a,b),'before_path':str(paths[0].relative_to(PROJECT)),'after_path':str(paths[1].relative_to(PROJECT))})
 dump(OUT/'snapshot_pairs.json',rows);dump(OUT/'downloads.json',downloads);dump(OUT/'licenses.json',licenses);dump(OUT/'mdn_sources.json',source_records)
 print(json.dumps({'actual_snapshot_pairs':len(rows),'same_geometry':sum(r['same_geometry'] for r in rows),'mdn_verified_source_versions':len(source_records),'source_html_css_edits':0}),flush=True)
if __name__=='__main__':main()
