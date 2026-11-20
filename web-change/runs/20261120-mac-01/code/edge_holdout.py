import csv
import json
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

from confirm import ROOT, extract, pair_diff


def edges_only(pixels, full):
    n=pixels.shape[1]
    return np.c_[pixels,full[:,n+27:-6]]


def main():
    started=time.perf_counter()
    known=set()
    dev,devdata,_=extract("first_rows.json",known)
    inspected,_,_=extract("confirm_rows.json",known)
    holdout,data,skipped=extract("edge_holdout_rows.json",known)
    assert len(dev)==84 and len(holdout)>120
    y=np.asarray([row["label"] for row in dev])
    actual=np.asarray([row["label"] for row in holdout])
    groups=np.asarray([row["domain"] for row in holdout])
    metrics={}
    outputs={}
    for name,p,q in [
            ("pixel",devdata[0],data[0]),
            ("pixel_hog",devdata[2],data[2]),
            ("edge",edges_only(*devdata[:2]),edges_only(*data[:2]))]:
        clf=ExtraTreesClassifier(n_estimators=300,max_features=.7,class_weight="balanced",random_state=13,n_jobs=1)
        pred=clf.fit(p,y).predict(q)
        outputs[name]=pred
        metrics[name]={"accuracy":accuracy_score(actual,pred),"macro_f1":f1_score(actual,pred,average="macro")}
        for row,result in zip(holdout,pred):
            row[name]=result
    ci={name:pair_diff(groups,outputs[name],outputs["edge"],actual) for name in ("pixel","pixel_hog")}
    result={"dev":len(dev),"holdout":len(holdout),"diagnostic_pairs_not_used":len(inspected),"metrics":metrics,
            "paired_domain_bootstrap_95_ci":ci,"exclusions":skipped,"seconds":time.perf_counter()-started}
    (ROOT/"edge_holdout_summary.json").write_text(json.dumps(result,indent=2))
    with (ROOT/"edge_holdout_predictions.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(holdout[0]))
        writer.writeheader()
        writer.writerows(holdout)
    print(json.dumps(result),flush=True)

if __name__=="__main__":main()
