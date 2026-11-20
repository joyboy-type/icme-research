from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import requests
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
GROUP_NAMES = {
    "text": "typography", "font_size": "typography", "font_weight": "typography",
    "letter_spacing": "typography", "line_height": "typography",
    "border": "boundary", "rounded": "boundary",
    "spacing": "layout", "justify": "layout", "position": "layout",
    "color": "photometric", "opacity": "photometric", "gradient": "photometric",
}

cv2.setNumThreads(1)
start = time.perf_counter()


def get_image(item, suffix):
    name = item["id"] + "_" + suffix + ".img"
    path = ROOT/"samples"/name
    if not path.exists():
        response = requests.get(item["image_" + suffix]["src"], timeout=16)
        response.raise_for_status()
        raw = response.content
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        path.write_bytes(raw)
    return np.asarray(Image.open(path).convert("RGB"))


def get_pair(item):
    try:
        return item, get_image(item,"before"), get_image(item,"after"), None
    except Exception as e:
        return item, None, None, str(e)


def summarize(a, quantiles=(.1,.25,.5,.75,.9)):
    flat = np.asarray(a).ravel()
    if not len(flat):
        return [0.] * (2+len(quantiles))
    return [float(flat.mean()), float(flat.std()), *np.quantile(flat,quantiles)]


def projection_stats(values):
    out = []
    for axis in (0, 1):
        p = values.mean(axis=axis)
        out.extend(summarize(p))
        out.append(float(np.count_nonzero(p)/max(1,p.size)))
        if len(p)>1:
            out.extend(summarize(np.abs(np.diff(p))))
        else:
            out.extend(summarize([]))
    return out


def features(a,b):
    if a.shape!=b.shape:
        raise ValueError(f"Screenshot size mismatch {a.shape} and {b.shape}")
    h,w=a.shape[:2]
    # Restrict all methods to identical pixel support; no target coordinates from annotation.
    absolute=np.abs(a.astype(np.float32)-b.astype(np.float32))
    delta=absolute.max(axis=2)
    ys,xs=np.where(delta>4)
    if len(xs)==0:
        ys,xs=np.where(delta>0)
    if len(xs)==0:
        raise ValueError("No changed pixels in has-diff data")
    y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    pad=max(4,round(.12*min(y1-y0,x1-x0)))
    crop=(slice(max(0,y0-pad),min(h,y1+pad)),slice(max(0,x0-pad),min(w,x1+pad)))
    ca=a[crop];cb=b[crop];d=delta[crop]
    m=d>4
    if not m.any():
        m=d>0
    scale=[(y1-y0)/h,(x1-x0)/w,np.sqrt(len(xs)/(h*w)),len(xs)/((y1-y0)*(x1-x0))]
    base=scale+summarize(d/m.size)+summarize(d[m]/255)
    base+=projection_stats(m.astype(np.float32))
    for channel in range(3):
        base+=summarize(absolute[crop][:,:,channel][m] / 255)
    patch=cv2.resize(d,(16,16),interpolation=cv2.INTER_AREA).ravel()/255
    geometry=[]
    lab_a=cv2.cvtColor(ca,cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_b=cv2.cvtColor(cb,cv2.COLOR_RGB2LAB).astype(np.float32)
    signed=(lab_b-lab_a)/255
    for channel in range(3):
        plane=signed[:,:,channel]
        geometry+=summarize(plane[m])
        geometry += [float((plane[m]>0.01).mean()),float((plane[m]<-0.01).mean())]
    ga=cv2.cvtColor(ca,cv2.COLOR_RGB2GRAY)
    gb=cv2.cvtColor(cb,cv2.COLOR_RGB2GRAY)
    ea=cv2.Canny(ga,40,100)>0
    eb=cv2.Canny(gb,40,100)>0
    new=eb & ~ea; old=ea & ~eb
    geometry += [float(ea.mean()),float(eb.mean()),float(new.mean()),float(old.mean()),float((ea&eb).sum()/max(1,(ea|eb).sum()))]
    geometry += projection_stats(new.astype(np.float32))+projection_stats(old.astype(np.float32))
    ha,wa=ga.shape
    if min(ha,wa)>=2:
        a32=ga.astype(np.float32)
        b32=gb.astype(np.float32)
        if max(ha,wa)>512:
            shape=(max(2,round(512*wa/max(ha,wa))),max(2,round(512*ha/max(ha,wa))))
            a32=cv2.resize(a32,shape);b32=cv2.resize(b32,shape)
        (dx,dy),strength=cv2.phaseCorrelate(a32,b32)
        if not np.isfinite(dx+dy+strength):
            dx=dy=strength=0.0
        geometry += [dx/a32.shape[1],dy/a32.shape[0],strength]
        warped=cv2.warpAffine(a32,np.array([[1,0,dx],[0,1,dy]],dtype=np.float32),(a32.shape[1],a32.shape[0]),borderMode=cv2.BORDER_REPLICATE)
        raw_error=float(np.abs(a32-b32).mean())
        aligned_error=float(np.abs(warped-b32).mean())
        geometry += [raw_error/255,aligned_error/255,(raw_error-aligned_error)/max(raw_error,1)]
    else:
        geometry += [0.0]*6
    return np.r_[base,patch],np.r_[base,patch,geometry]


def main():
    data=[p["row"] for p in json.loads((ROOT/"first_rows.json").read_text())["rows"]]
    (ROOT/"samples").mkdir(exist_ok=True)
    rows=[]
    simple=[];differential=[]
    failures=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n,(row,a,b,error) in enumerate(pool.map(get_pair,data),1):
            if error:
                failures.append([row["id"],error])
            else:
                try:
                    x,y=features(a,b)
                    simple.append(x);differential.append(y)
                    label=row["mutation_types"][0].removeprefix("mutate_")
                    rows.append(dict(id=row["id"],difficulty=row["difficulty"],domain=row["domain"],label=label,group=GROUP_NAMES[label]))
                except ValueError as e:
                    failures.append([row["id"],str(e)])
            if n%20==0:
                print("features",n,len(rows),len(failures),flush=True)
    assert len(rows)>=75,failures
    domains=np.array([r["domain"] for r in rows])
    result={}
    for target in ("label","group"):
        truth=np.array([r[target] for r in rows])
        scores={}
        for name,x in [("pixel_baseline",simple),("differential",differential)]:
            x=np.asarray(x)
            models={
                "trees": ExtraTreesClassifier(n_estimators=300,max_features=.7,class_weight="balanced",random_state=13,n_jobs=1),
                "svm":make_pipeline(StandardScaler(),SVC(C=3,class_weight="balanced"))
            }
            for model,estimator in models.items():
                yhat=np.empty(len(truth),dtype=truth.dtype)
                for tr,te in LeaveOneGroupOut().split(x, truth, domains):
                    fit=clone(estimator).fit(x[tr],truth[tr])
                    yhat[te]=fit.predict(x[te])
                key=name+"_"+model+"_"+target
                for row,answer in zip(rows,yhat):
                    row[key]=answer
                scores[name+"_"+model]=[float(accuracy_score(truth,yhat)),float(f1_score(truth,yhat,average="macro"))]
        result[target]=scores
    with (ROOT/"paired_results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report=dict(n=len(rows),domain_count=len(set(domains)),failures=failures,metrics=result,elapsed_s=time.perf_counter()-start)
    (ROOT/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
