"""Fit on train families, calibrate on development controls, then invoke sealed confirmation separately."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
from time import perf_counter
import argparse,csv,json,joblib,platform,sys
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from evaluation_utils import role_masks,detection_metrics,file_hashes,check_lock
from registration import calibration_threshold
RUN=Path(__file__).resolve().parents[1];PROJECT=RUN.parents[1];DATA=PROJECT/'data/template-probes-01';OUT=RUN/'outputs'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def estimator():return ExtraTreesClassifier(**read(RUN/'evaluation_config.json')['model'])
def guard():
 for p,h in read(RUN/'protocol.json')['word_guard'].items():assert sha256(Path(p).read_bytes()).hexdigest()==h,'Frozen Word changed'
def write_csv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def classification_metric(y,p,labels):
 return {'n':len(y),'accuracy':float(accuracy_score(y,p)),'macro_f1_fixed13':float(f1_score(y,p,labels=labels,average='macro',zero_division=0)),
  'macro_f1_fixed12_excluding_untrained_justify':float(f1_score(y,p,labels=[v for v in labels if v!='justify'],average='macro',zero_division=0)),
  'per_class':classification_report(y,p,labels=labels,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y,p,labels=labels).tolist()}
def classification_ci(y,base,other,groups,labels):
 names=sorted(set(groups));lookup={v:i for i,v in enumerate(labels)};yi=np.array([lookup[v] for v in y]);k=len(labels);counts=[]
 for p in [base,other]:
  pi=np.array([lookup[v] for v in p]);counts.append(np.stack([np.bincount(k*yi[groups==g]+pi[groups==g],minlength=k*k).reshape(k,k) for g in names]))
 def f1(cm):
  den=cm.sum(0)+cm.sum(1);return np.divide(2*np.diag(cm),den,out=np.zeros(k),where=den!=0).mean()
 rng=np.random.default_rng(71);delta=[]
 for _ in range(2000):
  w=np.bincount(rng.integers(len(names),size=len(names)),minlength=len(names));a,b=[np.tensordot(w,c,axes=1) for c in counts];delta.append(f1(b)-f1(a))
 return {'delta_macro_f1':float(f1(counts[1].sum(0))-f1(counts[0].sum(0))),'ci95':np.quantile(delta,[.025,.975]).tolist(),'families':len(names),'resamples':2000,'conditional_on_frozen_models':True}
def rate_ci(base,other,groups):
 names=sorted(set(groups));size=np.array([(groups==g).sum() for g in names]);sums=np.array([(other[groups==g]-base[groups==g]).sum() for g in names]);rng=np.random.default_rng(71);values=[]
 for _ in range(2000):
  w=np.bincount(rng.integers(len(names),size=len(names)),minlength=len(names));values.append(float(w@sums/(w@size)))
 return {'delta_rate':float(np.mean(other-base)),'ci95':np.quantile(values,[.025,.975]).tolist(),'families':len(names),'resamples':2000,'conditional_on_frozen_models_and_thresholds':True}
def inputs():
 crows=read(OUT/'classification_index.json');drows=read(OUT/'detection_index.json');cf=np.load(OUT/'classification_features.npz');df=np.load(OUT/'detection_features.npz')
 protocol=read(RUN/'protocol.json');cfg=read(RUN/'evaluation_config.json')
 assert all(r['split']==protocol['family_split'][r['family']] for r in crows+drows)
 assert len({(r['baseline_id'],r['condition']) for r in drows if not r['positive']})==sum(not r['positive'] for r in drows)
 for split in ['train','development','confirmation']:
  assert {r['family'] for r in crows if r['split']==split}=={g for g,s in protocol['family_split'].items() if s==split}
 return crows,drows,cf,df,protocol,cfg
def fit():
 assert not (RUN/'confirmation_lock.json').exists(),'Already locked; do not refit'
 guard();started=perf_counter();crows,drows,cf,df,protocol,cfg=inputs();labels=protocol['generation']['labels']
 train=np.array([r['split']=='train' for r in crows]);dev=np.array([r['split']=='development' for r in crows]);y=np.array([r['label'] for r in crows]);classifiers={};devscores={}
 for method in protocol['evaluation']['classification']:
  classifiers[method]=estimator().fit(cf[method][train],y[train]);devscores[method]=classification_metric(y[dev],classifiers[method].predict(cf[method][dev]),labels)
 fitmask,calmask,_=role_masks(drows,cfg['calibration_negative_conditions'],cfg['positive_detection_conditions']);dy=np.array([r['positive'] for r in drows]);conditions=np.array([r['condition'] for r in drows]);model=estimator().fit(df['x'][fitmask],dy[fitmask]);calprob=model.predict_proba(df['x'][calmask])[:,list(model.classes_).index(1)]
 methods={};thresholds={};devdrows=[r for r in drows if r['split']=='development'];devmask=np.array([r['split']=='development' for r in drows]);dev_predictions=[];dmetrics={}
 for j,name in enumerate(['whole_area_blur8','local_max32','learned_local_gate']):
  cal=calprob if name=='learned_local_gate' else df['scalar'][calmask,j]
  by={c:cal[conditions[calmask]==c] for c in cfg['calibration_negative_conditions']};thresholds[name]=calibration_threshold(by)
  scores=model.predict_proba(df['x'][devmask])[:,list(model.classes_).index(1)] if name=='learned_local_gate' else df['scalar'][devmask,j]
  dmetrics[name]=detection_metrics(devdrows,scores>thresholds[name])
  for r,s in zip(devdrows,scores):dev_predictions.append({**{k:r[k] for k in ['id','family','split','label','positive','condition','baseline_id']},'method':name,'score':float(s),'threshold':thresholds[name],'detected':int(s>thresholds[name])})
 external_path=RUN.parent/'20261120-mac-04/outputs/frozen_gate.joblib';external=joblib.load(external_path)
 assert external['feature_names']==read(OUT/'feature_names.json')
 frozen={'classifiers':classifiers,'detector':model,'external_detector':external,'thresholds':thresholds,'class_labels':labels}
 joblib.dump(frozen,OUT/'frozen_models.joblib');dump(OUT/'thresholds.json',thresholds);write_csv(OUT/'development_detection_predictions.csv',dev_predictions)
 dump(OUT/'development_summary.json',{'classification':devscores,'detection':dmetrics,'fit_positive_rows':int(dy[fitmask].sum()),'fit_negative_rows':int((1-dy[fitmask]).sum()),'calibration_negative_rows':int(calmask.sum()),'calibration_unique_baselines':len({r['baseline_id'] for r,m in zip(drows,calmask) if m}),'zero_train_support':['justify'],'no_selection_from_development_metrics':True,'seconds':perf_counter()-started})
 paths=list((RUN/'code').glob('*.py'))+list((RUN/'code').glob('*.mjs'))+[RUN/'protocol.json',RUN/'evaluation_config.json',RUN/'source_specs.json',DATA/'manifest.json',DATA/'baselines.json',DATA/'sources.json',DATA/'exclusions.json',OUT/'classification_features.npz',OUT/'classification_index.json',OUT/'detection_features.npz',OUT/'detection_index.json',OUT/'feature_names.json',OUT/'classification_extraction.json',OUT/'detection_extraction.json',OUT/'provenance_verification.json',OUT/'dataset_audit.json',OUT/'frozen_models.joblib',OUT/'thresholds.json',external_path]
 dump(RUN/'confirmation_lock.json',{'at_utc':now(),'hashes':file_hashes(paths,PROJECT),'policy':'Models, thresholds, methods, conditions, source-family split frozen. Confirmation scores unopened.','methods':protocol['evaluation']['classification'],'detectors':['whole_area_blur8','local_max32','learned_local_gate','external_run04_gate'],'external_gate_scope':'Frozen screenshot-group-development method; diagnostic transfer only','zero_training_support_class':'justify','python':sys.version,'platform':platform.platform()})
 guard();print('FIT COMPLETE; CONFIRMATION LOCKED',json.dumps(thresholds),flush=True)
def confirmation():
 lock=read(RUN/'confirmation_lock.json');check_lock(lock['hashes'],PROJECT);assert not (RUN/'confirmation_opened.json').exists(),'Confirmation already opened'
 guard();started=perf_counter();crows,drows,cf,df,protocol,cfg=inputs();frozen=joblib.load(OUT/'frozen_models.joblib');labels=frozen['class_labels']
 dump(RUN/'confirmation_opened.json',{'at_utc':now(),'lock_sha256':sha256((RUN/'confirmation_lock.json').read_bytes()).hexdigest()})
 cmask=np.array([r['split']=='confirmation' for r in crows]);rows=[r for r in crows if r['split']=='confirmation'];y=np.array([r['label'] for r in rows]);groups=np.array([r['family'] for r in rows]);classpred={};classmetrics={};cp=[]
 for name,m in frozen['classifiers'].items():
  pred=m.predict(cf[name][cmask]);classpred[name]=pred;classmetrics[name]=classification_metric(y,pred,labels)
  for r,p in zip(rows,pred):cp.append({**r,'method':name,'prediction':p})
 write_csv(OUT/'confirmation_classification_predictions.csv',cp)
 dmask=np.array([r['split']=='confirmation' for r in drows]);dr=[r for r in drows if r['split']=='confirmation'];dp=[];dm={};decisions={};cleangates={}
 for j,name in enumerate(lock['detectors']):
  if name=='external_run04_gate':
   external=frozen['external_detector'];m=external['model'];scores=m.predict_proba(df['x'][dmask])[:,list(m.classes_).index(1)];threshold=external['threshold']
  elif name=='learned_local_gate':
   m=frozen['detector'];scores=m.predict_proba(df['x'][dmask])[:,list(m.classes_).index(1)];threshold=frozen['thresholds'][name]
  else:scores=df['scalar'][dmask,j];threshold=frozen['thresholds'][name]
  pred=scores>threshold;decisions[name]=pred.astype(float);dm[name]={'threshold':threshold,**detection_metrics(dr,pred)};cleangates[name]={r['pair_id']:bool(p) for r,p in zip(dr,pred) if r['positive'] and r['condition']=='clean'}
  for r,s,p in zip(dr,scores,pred):dp.append({**{k:r[k] for k in ['id','family','split','label','positive','condition','baseline_id']},'pair_id':r.get('pair_id',''),'method':name,'score':float(s),'threshold':threshold,'detected':int(p)})
 write_csv(OUT/'confirmation_detection_predictions.csv',dp)
 boot={};dg=np.array([r['family'] for r in dr])
 for name in lock['detectors'][1:]:
  boot[name]={}
  for positive,kind,conds in [(1,'recall',cfg['positive_detection_conditions']),(0,'fpr',cfg['negative_conditions'])]:
   for c in conds:
    mask=np.array([r['positive']==positive and r['condition']==c for r in dr]);boot[name][kind+'_'+c]=rate_ci(decisions['whole_area_blur8'][mask],decisions[name][mask],dg[mask])
 gated={}
 for detector,gates in cleangates.items():
  gated[detector]={}
  for name,p in classpred.items():
   accepted=np.array([gates[r['id']] for r in rows]);gp=p.copy();gp[~accepted]='no_diff'
   gated[detector][name]={'macro_f1_fixed13_rejections_are_errors':float(f1_score(y,gp,labels=labels,average='macro',zero_division=0)),'accuracy':float(accuracy_score(y,gp))}
 summary={'scope':protocol['scope'],'confirmation_pairs':len(rows),'confirmation_families':len(set(groups)),'confirmation_control_baselines':len({r['baseline_id'] for r in dr if not r['positive']}),'labels':labels,'zero_train_support':['justify'],'classification':classmetrics,'classification_family_bootstrap_vs_pixel':{name:classification_ci(y,classpred['pixel_ET'],p,groups,labels) for name,p in classpred.items() if name!='pixel_ET'},'detection':dm,'detection_family_bootstrap_vs_whole_area':boot,'gated_classification_clean':gated,'limitations':['Sanitized framework probes, not real production sites.','Only6 confirmation families; bootstrap conditional on fitted models and thresholds.','Calibration has6 views from2 families. Observed FPR is not population guarantee.','justify has0 training examples; position has5.','Repeated scroll views and CSS levels are dependent; family is resampling unit.','Run04 external detector is a separate transfer diagnostic.'],'seconds':perf_counter()-started}
 dump(OUT/'confirmation_summary.json',summary);check_lock(lock['hashes'],PROJECT);guard();print('CONFIRMATION COMPLETE',len(rows),'pairs',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','confirmation']);a=p.parse_args();(fit if a.stage=='fit' else confirmation)()
