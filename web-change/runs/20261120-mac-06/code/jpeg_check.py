"""One predeclared development preprocessing check; no confirmation examples scored."""
from pathlib import Path
from hashlib import sha256
from time import perf_counter
from datetime import datetime,timezone
import sys,json,csv,joblib
import cv2,numpy as np
from PIL import Image
RUN=Path(__file__).resolve().parents[1];PREV=RUN.parent/'20261120-mac-05';PROJECT=RUN.parents[1];DATA=PROJECT/'data/template-probes-01';OUT=RUN/'outputs'
sys.path.insert(0,str(PREV/'code'))
from extract import variant
from registration import estimate_translation,translate,calibration_threshold
from local_features import describe
from evaluation_utils import detection_metrics,file_hashes,check_lock
ALLOWED={'materialize','uikit','bootstrap','milligram','spectre','chota'}
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def select_development(rows):
 chosen={}
 for r in sorted((r for r in rows if r['split']=='development'),key=lambda r:({1:0,0:1,2:2}[r['level']],r['baseline_id'],r['id'])):chosen.setdefault((r['family'],r['label']),r)
 return [chosen[k] for k in sorted(chosen)]
def pair_inputs(a,b,condition,pipeline):
 received=variant(b,condition)
 if pipeline=='raw':return a,received
 if pipeline=='canonical90':return variant(a,'jpeg90'),variant(received,'jpeg90')
 if pipeline=='matched_capture':return variant(a,condition),received
 raise ValueError(pipeline)
def load(r,field):
 assert r['family'] in ALLOWED and r['split'] in ['train','development'];return np.asarray(Image.open(PROJECT/r[field]).convert('RGB'))
def main():
 assert not (OUT/'summary.json').exists(),'One run only'
 cfg=read(RUN/'protocol.json');guard=cfg['word_guard'];check_lock(read(PREV/'confirmation_lock.json')['hashes'],PROJECT)
 for p,h in guard.items():assert sha256(Path(p).read_bytes()).hexdigest()==h
 paths=[RUN/'protocol.json',*sorted((RUN/'code').glob('*.*')),PREV/'confirmation_lock.json',PREV/'outputs/frozen_models.joblib',PREV/'outputs/detection_features.npz',PREV/'outputs/detection_index.json',DATA/'manifest.json',DATA/'baselines.json']
 dump(RUN/'design_lock.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'hashes':file_hashes(paths,PROJECT),'no_new_confirmation':True})
 cv2.setNumThreads(1);started=perf_counter();manifest=read(DATA/'manifest.json');positives=select_development(manifest);assert all(r['family'] in ['spectre','chota'] for r in positives)
 bases=[r for r in read(DATA/'baselines.json') if r['split'] in ['train','development']];assert len(bases)==16 and all(r['family'] in ALLOWED for r in bases)
 cohort=[]
 for r in bases:
  for c in cfg['jpeg_experiment']['raw_controls']:cohort.append((r,0,c))
 for r in positives:
  for c in cfg['jpeg_experiment']['positive_conditions']:cohort.append((r,1,c))
 rawindex=read(PREV/'outputs/detection_index.json');bank=np.load(PREV/'outputs/detection_features.npz');lookup={(r.get('pair_id',r['baseline_id']),r['positive'],r['condition']):i for i,r in enumerate(rawindex) if r['split'] in ['train','development']}
 records=[];features=[];scalar=[];inputdiff=[]
 def add(r,pos,c,pipeline):
  record={'id':r['id'],'baseline_id':r.get('baseline_id',r['id']),'family':r['family'],'split':r['split'],'label':r['label'] if pos else 'no_diff','positive':pos,'condition':c,'pipeline':pipeline}
  if pipeline=='raw':
   i=lookup[(r['id'],pos,c)];v=bank['x'][i];s=bank['scalar'][i,1];diff=None
  else:
   a=load(r,'before_path');b=load(r,'after_path') if pos else a;a,b=pair_inputs(a,b,c,pipeline);diff=float((np.abs(a.astype(np.int16)-b.astype(np.int16)).max(2)>4).mean())
   alignment=estimate_translation(a,b,'global_phase');v,n,sc=describe(a,translate(b,*[-k for k in alignment['shift']]));s=sc['max_local32']
  records.append(record);features.append(v);scalar.append(s);inputdiff.append(diff)
 for pipeline in ['raw','canonical90']:
  for r,pos,c in cohort:add(r,pos,c,pipeline)
  print('features pipeline',pipeline,'rows',len(records),'seconds',round(perf_counter()-started,1),flush=True)
 for r in bases:
  for c in ['jpeg90','jpeg95','jpeg98']:add(r,0,c,'matched_capture')
 for r in positives:add(r,1,'jpeg95','matched_capture')
 x=np.asarray(features);scalar=np.asarray(scalar);frozen=joblib.load(PREV/'outputs/frozen_models.joblib');model=frozen['detector'];learned=model.predict_proba(x)[:,list(model.classes_).index(1)];oldthreshold=frozen['thresholds']['learned_local_gate'];summary={};output=[]
 for pipeline in ['raw','canonical90','matched_capture']:
  tr=np.array([r['pipeline']==pipeline and r['split']=='train' and r['positive']==0 for r in records]);te=np.array([r['pipeline']==pipeline and r['split']=='development' for r in records]);dr=[r for r,m in zip(records,te) if m]
  for method,scores in [('local_max32',scalar),('frozen_run05_learned_gate',learned)]:
   if method=='frozen_run05_learned_gate':threshold=oldthreshold
   elif pipeline=='matched_capture':threshold=summary['raw'][method]['threshold']
   else:threshold=calibration_threshold({c:scores[tr&np.array([r['condition']==c for r in records])] for c in cfg['jpeg_experiment']['calibration_conditions']})
   result={'threshold':threshold,**detection_metrics(dr,scores[te]>threshold)};summary.setdefault(pipeline,{})[method]=result
   for r,s in zip(dr,scores[te]):output.append({**r,'method':method,'score':float(s),'threshold':threshold,'detected':int(s>threshold)})
 verdict={}
 for method in ['local_max32','frozen_run05_learned_gate']:
  raw=summary['raw'][method];can=summary['canonical90'][method];wr=max(v['rate'] for v in raw['fpr'].values());wc=max(v['rate'] for v in can['fpr'].values());safe=all(can['recall'][c]['rate']>=raw['recall'][c]['rate'] for c in ['clean','jpeg95'])
  verdict[method]={'raw_worst_development_fpr':wr,'canonical_worst_development_fpr':wc,'preserves_clean_and_jpeg95_recall':safe,'predeclared_signal':wc<wr and safe}
 matchedneg=[d for r,d in zip(records,inputdiff) if r['pipeline']=='matched_capture' and not r['positive']];assert all(d==0 for d in matchedneg)
 np.savez_compressed(OUT/'features.npz',x=x,local_max32=scalar,learned_scores=learned);dump(OUT/'feature_index.json',records);dump(OUT/'selection.json',[{k:r[k] for k in ['id','family','label','level','baseline_id']} for r in positives])
 with (OUT/'decisions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
 dump(OUT/'summary.json',{'scope':'Development-only, one predeclared canonicalization candidate. No confirmation metrics used.','development_positive_pairs':len(positives),'train_negative_baselines':10,'development_negative_baselines':6,'development_families':2,'metrics':summary,'decision':verdict,'matched_capture_negative_pixel_difference_all_zero':True,'matched_capture_diagnostic_warning':'Same encoding on identical frames is a construction sanity check. It does not repair singly encoded incoming pairs.','seconds':perf_counter()-started,'no_iteration':True})
 check_lock(read(RUN/'design_lock.json')['hashes'],PROJECT)
 for p,h in guard.items():assert sha256(Path(p).read_bytes()).hexdigest()==h
 print('JPEG CHECK COMPLETE',json.dumps(verdict),flush=True)
if __name__=='__main__':main()
