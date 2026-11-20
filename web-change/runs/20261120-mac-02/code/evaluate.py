"""Development selection, then separately invoked locked confirmation."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
from time import perf_counter
import argparse,csv,json,platform,sys
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from core import false_positive_threshold
from prepare import RUN,NEW,PROJECT,dump

LABELS=json.loads((RUN/'protocol.json').read_text())['labels']
METHODS={'W0':('W0','trees'),'W1':('W1','trees'),'W2':('directional','trees'),
         'W3_merged':('merged','trees'),'W3_symmetric':('symmetric','trees'),
         'W3_new_only':('new_only','trees'),'W3_old_only':('old_only','trees'),
         'W4_dino_trees':('dino','trees'),'W4_dino_svm':('dino','svm'),
         'W4_dino_pixel_trees':('dino_pixel','trees'),'W4_dino_pixel_svm':('dino_pixel','svm')}

def estimator(family):
    if family=='trees':return ExtraTreesClassifier(n_estimators=300,max_features=.7,class_weight='balanced',random_state=13,n_jobs=1)
    return make_pipeline(StandardScaler(),SVC(C=3,class_weight='balanced'))

def metric(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,labels=LABELS,average='macro',zero_division=0)),
            'per_class':classification_report(y,p,labels=LABELS,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y,p,labels=LABELS).tolist()}

def ci(y,baseline,proposed,groups):
    names=sorted(set(groups));counts=[]
    lookup={x:i for i,x in enumerate(LABELS)};yi=np.array([lookup[v] for v in y])
    for p in [baseline,proposed]:
        pi=np.array([lookup[v] for v in p]);counts.append(np.stack([np.bincount(13*yi[groups==g]+pi[groups==g],minlength=169).reshape(13,13) for g in names]))
    rng=np.random.default_rng(71);diff=[]
    def f1(cm):
        den=cm.sum(axis=0)+cm.sum(axis=1)
        return np.divide(2*np.diag(cm),den,out=np.zeros(13),where=den!=0).mean()
    for _ in range(2000):
        w=np.bincount(rng.integers(len(names),size=len(names)),minlength=len(names))
        a,b=[np.tensordot(w,c,axes=1) for c in counts];diff.append(f1(b)-f1(a))
    return {'delta_macro_f1':float(f1_score(y,proposed,labels=LABELS,average='macro',zero_division=0)-f1_score(y,baseline,labels=LABELS,average='macro',zero_division=0)),
            'ci95':np.quantile(diff,[.025,.975]).tolist(),'groups':len(names),'replicates':2000,'group_type':'near_duplicate_proxy_not_website'}

def fingerprint():
    paths=[RUN/'protocol.json',NEW/'manifest.json',NEW/'labels.sealed.json',RUN/'outputs/features.npz',RUN/'outputs/feature_index.json',RUN/'outputs/negative_scores.json']+sorted((RUN/'code').glob('*.py'))
    paths += [PROJECT.parents[1]/p for p in json.loads((RUN/'protocol.json').read_text())['weights']]
    return {str(p.relative_to(RUN) if p.is_relative_to(RUN) else p):sha256(p.read_bytes()).hexdigest() for p in paths}

def write_predictions(path,rows,truth,predictions,latency=None):
    fields=['run_id','pair_id','split','source_group','source_url','before_sha256','after_sha256','label','crop_xyxy','method','output_raw','label_pred','fail_reason','elapsed_ms']
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for method,values in predictions.items():
            for r,y,p in zip(rows,truth,values):
                writer.writerow({'run_id':RUN.name,'pair_id':r['id'],'split':r['split'],'source_group':r['source_group'],'source_url':r['source_url'],
                 'before_sha256':r['before_sha256'],'after_sha256':r['after_sha256'],'label':y,'crop_xyxy':json.dumps(r['crop_xyxy']),
                 'method':method,'output_raw':p,'label_pred':p,'fail_reason':'','elapsed_ms':latency.get(method,'') if latency else ''})

def development():
    index=json.loads((RUN/'outputs/feature_index.json').read_text());features=np.load(RUN/'outputs/features.npz')
    labels=json.loads((NEW/'labels.sealed.json').read_text())
    tr=np.array([r['split']=='development' for r in index]);rows=[r for r in index if r['split']=='development']
    y=np.array([labels[r['id']]['label'] for r in rows]);groups=np.array([r['source_group'] for r in rows])
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261120).split(np.zeros(len(y)),y,groups))
    for a,b in folds:assert set(groups[a]).isdisjoint(groups[b])
    results={};predictions={}
    for name,(kind,family) in METHODS.items():
        x=features[kind][tr];pred=np.empty(len(y),dtype=object)
        t=perf_counter()
        for a,b in folds:pred[b]=estimator(family).fit(x[a],y[a]).predict(x[b])
        predictions[name]=pred;results[name]=metric(y,pred)
        print('development',name,round(results[name]['macro_f1'],4),'seconds',round(perf_counter()-t,1),flush=True)
    selected=sorted([k for k in METHODS if k.startswith('W4')],key=lambda k:(-results[k]['macro_f1'],k))[0]
    negative=json.loads((RUN/'outputs/negative_scores.json').read_text())
    threshold=false_positive_threshold([r['score'] for r in negative if r['split']=='development'])
    dump(RUN/'outputs/development_summary.json',{'n':len(y),'groups':len(set(groups)),'folds':3,'metrics':results,'selected_W4':selected,'negative_threshold':threshold,
         'directionality_comparisons':{name:ci(y,predictions[name],predictions['W2'],groups) for name in ['W3_merged','W3_symmetric']}})
    write_predictions(RUN/'outputs/development_predictions.csv',rows,y,predictions)
    dump(RUN/'confirmation_lock.json',{'locked_at_utc':datetime.now(timezone.utc).isoformat(),'selected_W4':selected,'negative_threshold':threshold,
         'confirm_methods':['W0','W1','W2','W3_merged','W3_symmetric','W3_new_only','W3_old_only',selected],
         'hashes':fingerprint(),'claim_scope':'near-duplicate substitute grouping only; no true website identifiers',
         'policy':'No changes after fresh confirmation labels are read. Historical155 remains quarantined.'})
    print('CONFIRMATION LOCKED',selected,'threshold',threshold,flush=True)

def confirmation():
    lock=json.loads((RUN/'confirmation_lock.json').read_text());assert lock['hashes']==fingerprint(),'Locked inputs/code changed'
    dump(RUN/'confirmation_opened.json',{'opened_at_utc':datetime.now(timezone.utc).isoformat(),'lock_sha256':sha256((RUN/'confirmation_lock.json').read_bytes()).hexdigest()})
    index=json.loads((RUN/'outputs/feature_index.json').read_text());features=np.load(RUN/'outputs/features.npz');labels=json.loads((NEW/'labels.sealed.json').read_text())
    tr=np.array([r['split']=='development' for r in index]);te=np.array([r['split']=='confirmation' for r in index])
    dev=[r for r in index if r['split']=='development'];rows=[r for r in index if r['split']=='confirmation']
    assert set(r['source_group'] for r in dev).isdisjoint(r['source_group'] for r in rows)
    train=np.array([labels[r['id']]['label'] for r in dev]);y=np.array([labels[r['id']]['label'] for r in rows]);groups=np.array([r['source_group'] for r in rows])
    results={};predictions={};timings={}
    for name in lock['confirm_methods']:
        kind,family=METHODS[name];model=estimator(family)
        t=perf_counter();model.fit(features[kind][tr],train);fit_s=perf_counter()-t
        t=perf_counter();pred=model.predict(features[kind][te]);elapsed=(perf_counter()-t)*1000/len(y)
        timings[name]={'fit_seconds':fit_s,'classification_ms_amortized':elapsed}
        predictions[name]=pred;results[name]=metric(y,pred)
        print('confirmation',name,round(results[name]['macro_f1'],4),flush=True)
    comparisons={name:ci(y,predictions[name],predictions['W2'],groups) for name in predictions if name!='W2'}
    negative=json.loads((RUN/'outputs/negative_scores.json').read_text());neg_summary={}
    for split in ['development','confirmation']:
        neg_summary[split]={}
        kinds=sorted({r['condition'] for r in negative if r['split']==split})
        for kind in kinds:
            these=[r for r in negative if r['split']==split and r['condition']==kind]
            for r in these:r['threshold']=lock['negative_threshold'];r['flagged_change']=r['score']>r['threshold']
            neg_summary[split][kind]={'n':len(these),'fpr_raw_any_pixel_over4':float(np.mean([r['score']>0 for r in these])),
                'fpr_calibrated':float(np.mean([r['flagged_change'] for r in these]))}
    sensitivity={s:float(np.mean(features['positive_score'][mask]>lock['negative_threshold'])) for s,mask in [('development',tr),('confirmation',te)]}
    gated={}
    for name,pred in predictions.items():
        gated_pred=pred.copy();gated_pred[features['positive_score'][te]<=lock['negative_threshold']]='no_diff'
        gated[name]={'macro_f1_with_rejections_counted_as_errors':float(f1_score(y,gated_pred,labels=LABELS,average='macro',zero_division=0)),'accuracy':float(accuracy_score(y,gated_pred))}
    dump(RUN/'outputs/confirmation_summary.json',{'development':len(train),'confirmation':len(y),'groups':len(set(groups)),'labels':LABELS,'metrics':results,
         'paired_group_bootstrap':comparisons,'W5':{'threshold':lock['negative_threshold'],'negatives':neg_summary,'positive_sensitivity':sensitivity,'gated_classification':gated},
         'timing_classifiers':timings,'selected_W4':lock['selected_W4'],'scope':'Fresh examples, near-duplicate-proxy separation; not independently identified websites.'})
    dump(RUN/'outputs/negative_predictions.json',negative)
    write_predictions(RUN/'outputs/confirmation_predictions.csv',rows,y,predictions,{k:v['classification_ms_amortized'] for k,v in timings.items()})
    print('confirmation complete',len(y),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['development','confirmation']);a=parser.parse_args()
    (development if a.stage=='development' else confirmation)()
