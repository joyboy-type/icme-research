"""Download only pinned public version images and licenses, verifying Git blob bytes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from hashlib import sha256
from datetime import datetime,timezone
import json,requests,numpy as np
from PIL import Image
from pair_audit import git_blob_sha,pixel_audit
RUN=Path(__file__).resolve().parents[1];PROJECT=RUN.parents[1];DATA=PROJECT/'data/real-version-pairs-01';OUT=RUN/'outputs'
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def get(url,expected,dest):
 if dest.exists():
  raw=dest.read_bytes();assert git_blob_sha(raw)==expected;return {'url':url,'blob_sha':expected,'sha256':sha256(raw).hexdigest(),'bytes':len(raw),'cache':True}
 errors=[]
 for attempt in range(3):
  try:
   r=requests.get(url,timeout=25);r.raise_for_status();raw=r.content;assert git_blob_sha(raw)==expected,'Git blob mismatch';dest.write_bytes(raw)
   return {'url':url,'blob_sha':expected,'sha256':sha256(raw).hexdigest(),'bytes':len(raw),'cache':False}
  except Exception as e:errors.append(str(e))
 raise RuntimeError({'url':url,'errors':errors})
def main():
 started=perf_counter();cfg=json.loads((RUN/'protocol.json').read_text());specs=json.loads((RUN/'pair_specs.json').read_text());assert len(specs)==8 and len({s['repository'] for s in specs})==2
 for p,h in cfg['word_guard'].items():assert sha256(Path(p).read_bytes()).hexdigest()==h
 DATA.mkdir(exist_ok=True);(DATA/'blobs').mkdir(exist_ok=True);(DATA/'licenses').mkdir(exist_ok=True)
 dump(RUN/'design_lock.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'sha256':{str(p.relative_to(RUN)):sha256(p.read_bytes()).hexdigest() for p in [RUN/'protocol.json',RUN/'pair_specs.json',*sorted((RUN/'code').glob('*.py'))]},'policy':'Source audit only. No classifier, threshold, feature design or confirmation evaluation.'})
 jobs={}
 for s in specs:
  for version,key in [(s['parent_commit'],'before_blob_sha'),(s['commit'],'after_blob_sha')]:
   h=s[key];jobs.setdefault(h,(f"https://raw.githubusercontent.com/{s['repository']}/{version}/{s['path']}",h,DATA/'blobs'/(h+'.png')))
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures=[(h,pool.submit(get,*args)) for h,args in jobs.items()];downloads=[];errors=[]
  for h,f in futures:
   try:downloads.append(f.result())
   except Exception as e:errors.append({'blob_sha':h,'error':str(e)})
 dump(OUT/'downloads.json',downloads);dump(OUT/'download_errors.json',errors)
 assert not errors,errors
 licenses=[]
 for spec in cfg['licenses']:
  dest=DATA/'licenses'/(spec['repository'].replace('/','__')+'.txt');info=get(f"https://raw.githubusercontent.com/{spec['repository']}/{spec['commit']}/{spec['path']}",spec['blob_sha'],dest);licenses.append({**spec,**info,'local_path':str(dest.relative_to(PROJECT))})
 rows=[]
 for spec in specs:
  a,b=[np.asarray(Image.open(DATA/'blobs'/(spec[k]+'.png')).convert('RGB')) for k in ['before_blob_sha','after_blob_sha']];audit=pixel_audit(a,b)
  if spec.get('expected_no_change'):assert audit['same_geometry'] and audit['changed_pixels']==0
  rows.append({**spec,**audit,'before_path':str((DATA/'blobs'/(spec['before_blob_sha']+'.png')).relative_to(PROJECT)),'after_path':str((DATA/'blobs'/(spec['after_blob_sha']+'.png')).relative_to(PROJECT))})
 dump(DATA/'manifest.json',rows);dump(OUT/'pair_audit.json',rows);dump(OUT/'license_audit.json',licenses)
 summary={'scope':cfg['scope'],'pairs':len(rows),'project_groups':sorted({r['project_group'] for r in rows}),'candidate_modified_pairs':sum(not r.get('expected_no_change',False) for r in rows),'unchanged_controls':sum(r.get('expected_no_change',False) for r in rows),'same_geometry_pairs':sum(r['same_geometry'] for r in rows),'visible_changed_same_geometry_pairs':sum(r['same_geometry'] and r['changed_pixels']>0 for r in rows),'unique_downloaded_image_blobs':len(downloads),'download_bytes':sum(r['bytes'] for r in downloads),'licenses_verified':len(licenses),'seconds':perf_counter()-started,'pixel_resizing_applied':False,'synthetic_css_injection':False,'models_or_thresholds_evaluated':False,'word_unchanged_verified':True}
 dump(OUT/'summary.json',summary)
 for p,h in cfg['word_guard'].items():assert sha256(Path(p).read_bytes()).hexdigest()==h
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
