"""Nested grouped development validation for local residual features."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
from time import perf_counter
from collections import defaultdict
import csv,io,json,joblib
import cv2,numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from registration import estimate_translation,translate,calibration_threshold
from local_features import describe
from probe_metrics import class_mask

RUN=Path(__file__).resolve().parents[1];PROJECT=RUN.parents[1]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def model():return ExtraTreesClassifier(n_estimators=300,max_features=.7,class_weight='balanced',random_state=13,n_jobs=1)
def jpeg(image,quality):
    f=io.BytesIO();Image.fromarray(image).save(f,format='JPEG',quality=quality)
    return np.asarray(Image.open(io.BytesIO(f.getvalue())).convert('RGB'))
def variants(a):
    q=jpeg(a,95)
    return {'identity':a,'clean':a,'jpeg95':q,'shift_x1':translate(a,1,0),'shift_y1':translate(a,0,1),'shift_xy1':translate(a,1,1),'jpeg95_shift_x1':translate(q,1,0),'jpeg90':jpeg(a,90),'jpeg98':jpeg(a,98)}

def main():
    cv2.setNumThreads(1);out=RUN/'outputs';protocol=json.loads((RUN/'protocol.json').read_text());rows=json.loads((RUN/'inputs/development_manifest.json').read_text())
    assert len(rows)==240 and all(r['split']=='development' for r in rows)
    for path,h in protocol['word_guard'].items():assert sha256(Path(path).read_bytes()).hexdigest()==h
    started=perf_counter();data=[];records=[];scalar=[]
    dump(RUN/'design_lock.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha256((RUN/'protocol.json').read_bytes()).hexdigest(),'code_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in (RUN/'code').glob('*.py')}})
    for i,row in enumerate(rows):
        a,b=[np.asarray(Image.open(PROJECT/row[k+'_path']).convert('RGB')) for k in ['before','after']]
        for positive,image,conditions in [(0,a,protocol['training_negative_conditions']+protocol['ood_conditions']),(1,b,protocol['training_positive_conditions']+protocol['ood_conditions'])]:
            bank=variants(image)
            for condition in conditions:
                t=perf_counter();other=bank[condition];alignment=estimate_translation(a,other,'global_phase');aligned=translate(other,*[-v for v in alignment['shift']])
                features,names,scores=describe(a,aligned)
                data.append(features);scalar.append([scores[k] for k in ['area_blur8','max_local32','largest_component8']])
                records.append({'pair_id':row['id'],'group':row['source_group'],'original_index':i,'label':row['label'],'positive':positive,'condition':condition,'ood':condition in protocol['ood_conditions'],'feature_ms':(perf_counter()-t)*1000})
        if (i+1)%20==0:print('features',i+1,'seconds',round(perf_counter()-started,1),flush=True)
    x=np.asarray(data);sc=np.asarray(scalar);np.savez_compressed(out/'features.npz',x=x,scalar=sc);dump(out/'feature_records.json',records);dump(out/'feature_names.json',names)
    eligible=np.asarray([not r['ood'] for r in records]);oi=np.asarray([r['original_index'] for r in records]);y=np.asarray([r['positive'] for r in records]);condition=np.asarray([r['condition'] for r in records])
    labels=np.asarray([r['label'] for r in rows]);groups=np.asarray([r['source_group'] for r in rows]);assert len(set(groups))==240
    outer=list(StratifiedGroupKFold(3,shuffle=True,random_state=20261120).split(np.zeros(240),labels,groups))
    methods=['area_blur8','max_local32','largest_component8','learned_local'];prob={k:np.full(len(y),np.nan) for k in methods};pred={k:np.zeros(len(y),int) for k in methods};thresholds=[]
    for fold,(train,test) in enumerate(outer):
        assert set(groups[train]).isdisjoint(groups[test]);tr=np.isin(oi,train)&eligible;te=np.isin(oi,test)
        inner_scores=np.full(len(y),np.nan)
        inner=StratifiedGroupKFold(3,shuffle=True,random_state=20261120+fold+1)
        for fit,cal in inner.split(np.zeros(len(train)),labels[train],groups[train]):
            fit_mask=np.isin(oi,train[fit])&eligible;cal_mask=np.isin(oi,train[cal])&eligible
            inner_scores[cal_mask]=model().fit(x[fit_mask],y[fit_mask]).predict_proba(x[cal_mask])[:,1]
        fitted=model().fit(x[tr],y[tr]);learned=fitted.predict_proba(x[te])[:,1]
        for j,name in enumerate(methods):
            cal_scores=inner_scores if name=='learned_local' else sc[:,j]
            assert np.isfinite(cal_scores[tr]).all()
            by={c:cal_scores[tr&(y==0)&(condition==c)] for c in protocol['training_negative_conditions']}
            threshold=calibration_threshold(by)
            values=learned if name=='learned_local' else sc[te,j]
            prob[name][te]=values;pred[name][te]=(values>threshold).astype(int)
            thresholds.append({'fold':fold,'method':name,'threshold':threshold,'calibration':'inner OOF' if name=='learned_local' else 'outer training negatives'})
        print('outer fold',fold,'complete',round(perf_counter()-started,1),flush=True)
    summary={}
    for name in methods:
        entry={'recall':{},'fpr':{},'per_class_clean_recall':{}}
        for c in sorted(set(condition)):
            for positive,key in [(0,'fpr'),(1,'recall')]:
                mask=(condition==c)&(y==positive)
                if mask.any():entry[key][c]=float(pred[name][mask].mean())
        for label in sorted(set(labels)):
            mask=class_mask(records,label);entry['per_class_clean_recall'][label]=float(pred[name][mask].mean())
        entry['worst_seen_fpr']=max(entry['fpr'][c] for c in protocol['training_negative_conditions']);entry['worst_unseen_codec_fpr']=max(entry['fpr'][c] for c in protocol['ood_conditions']);summary[name]=entry
    fields=['pair_id','group','label','positive','condition','ood','method','score','detected']
    with (out/'predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for name in methods:
            for i,r in enumerate(records):w.writerow({**{k:r[k] for k in fields if k in r},'method':name,'score':prob[name][i],'detected':pred[name][i]})
    dump(out/'thresholds.json',thresholds);dump(out/'summary.json',{'scope':'Nested development OOF only; JPEG90/98 excluded from fitting and calibration','originals':240,'feature_rows':len(y),'features':x.shape[1],'metrics':summary,'seconds':perf_counter()-started})
    # Final portable gate uses full-development fitting and OOF negative calibration.
    full=model().fit(x[eligible],y[eligible]);by={c:prob['learned_local'][(y==0)&(condition==c)] for c in protocol['training_negative_conditions']}
    final_threshold=calibration_threshold(by)
    joblib.dump({'model':full,'threshold':final_threshold,'feature_names':names,'protocol_sha256':sha256((RUN/'protocol.json').read_bytes()).hexdigest()},out/'frozen_gate.joblib')
    dump(out/'freeze.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'threshold':final_threshold,'model_sha256':sha256((out/'frozen_gate.joblib').read_bytes()).hexdigest(),'selection':'Feature/model settings fixed before extraction; no independent confirmation evaluated.'})
    for path,h in protocol['word_guard'].items():assert sha256(Path(path).read_bytes()).hexdigest()==h
    print(json.dumps({'complete':True,'metrics':summary,'seconds':perf_counter()-started}),flush=True)

if __name__=='__main__':main()
