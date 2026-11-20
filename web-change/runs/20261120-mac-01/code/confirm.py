import csv
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import time

import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from paired import ROOT, get_pair, features, GROUP_NAMES

hog=cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
rng=np.random.default_rng(71)

def hog_comparison(a,b,base):
    maximum=np.abs(a.astype(np.int16)-b.astype(np.int16)).max(axis=2)
    yy,xx=np.nonzero(maximum>(4 if (maximum>4).any() else 0))
    pad=max(4,round(.12*min(np.ptp(yy)+1,np.ptp(xx)+1)))
    y0,y1=max(0,yy.min()-pad),min(a.shape[0],yy.max()+1+pad)
    x0,x1=max(0,xx.min()-pad),min(a.shape[1],xx.max()+1+pad)
    p,q=a[y0:y1,x0:x1],b[y0:y1,x0:x1]
    before=hog.compute(cv2.resize(cv2.cvtColor(p,cv2.COLOR_RGB2GRAY),(64,64))).ravel()
    after=hog.compute(cv2.resize(cv2.cvtColor(q,cv2.COLOR_RGB2GRAY),(64,64))).ravel()
    return np.r_[base,np.abs(after-before)]

def extract(file,seen):
    raw=[x["row"] for x in json.loads((ROOT/file).read_text())["rows"]]
    rows=[]
    observations=[]
    skipped=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,(info,a,b,error) in enumerate(pool.map(get_pair,raw),1):
            if error:
                skipped.append((info["id"],error))
                continue
            signature=sha256(a.tobytes()).hexdigest()
            if signature in seen:
                skipped.append((info["id"],"before image repeated"))
                continue
            assert a.shape==b.shape, (info["id"],a.shape,b.shape)
            seen.add(signature)
            simple, enriched=features(a,b)
            design=hog_comparison(a,b,simple)
            label=info["mutation_types"][0].removeprefix("mutate_")
            rows.append(dict(id=info["id"],label=label,group=GROUP_NAMES[label],domain=info["domain"],difficulty=info["difficulty"],before_sha256=signature))
            observations.append((simple,enriched,design))
            if i%30==0:
                print(file,i,"accepted",len(rows),"dropped",len(skipped),flush=True)
    return rows,[np.asarray(x) for x in zip(*observations)],skipped

def pair_diff(groups,a,b,truth):
    by_domain=[np.where(groups==g)[0] for g in np.unique(groups)]
    labels=np.unique(truth)
    values=[]
    for _ in range(1500):
        idx=np.concatenate([by_domain[i] for i in rng.integers(len(by_domain),size=len(by_domain))])
        values.append(f1_score(truth[idx],b[idx],average="macro",labels=labels,zero_division=0)-
                      f1_score(truth[idx],a[idx],average="macro",labels=labels,zero_division=0))
    return np.quantile(values,[.025,.975]).tolist()

def main():
    started=time.perf_counter()
    seen=set()
    dev,xdev,sdev=extract("first_rows.json",seen)
    check,xcheck,scheck=extract("confirm_rows.json",seen)
    assert len(check)>=120 and set(r["id"] for r in dev).isdisjoint(r["id"] for r in check)
    results={}
    intervals={}
    estimator={
        "trees":lambda:ExtraTreesClassifier(n_estimators=300,max_features=.7,class_weight="balanced",random_state=13,n_jobs=1),
        "svm":lambda:make_pipeline(StandardScaler(),SVC(C=3,class_weight="balanced")),
    }
    groups=np.array([r["domain"] for r in check])
    for metric in ("label","group"):
        train=np.array([x[metric] for x in dev])
        test=np.array([x[metric] for x in check])
        results[metric]={}
        preds={}
        for method,i in [("pixel",0),("enhanced",1),("pixel_hog",2)]:
            for family,init in estimator.items():
                model=init().fit(xdev[i],train)
                pred=model.predict(xcheck[i])
                key=method+"_"+family
                preds[key]=pred
                for r,v in zip(check,pred):
                    r[key+"_"+metric]=v
                results[metric][key]=dict(accuracy=accuracy_score(test,pred),macro_f1=f1_score(test,pred,average="macro"))
        intervals[metric]={}
        for family in estimator:
            for base in ("pixel","pixel_hog"):
                intervals[metric][family+"_"+base]=pair_diff(groups,preds[base+"_"+family],preds["enhanced_"+family],test)
    with (ROOT/"confirm_predictions.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(check[0]))
        w.writeheader()
        w.writerows(check)
    summary=dict(dev=len(dev),check=len(check),domains=len(set(groups)),metrics=results,paired_domain_bootstrap=intervals,dropped_dev=sdev,dropped_check=scheck,elapsed_s=time.perf_counter()-started)
    (ROOT/"confirm_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=="__main__":
    main()
