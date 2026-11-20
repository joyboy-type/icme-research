from pathlib import Path
import json,sys,subprocess,os,hashlib,time
from collections import Counter
import cv2
from gate import context_box,localized_decision,CONFIG

cv2.setNumThreads(2)
ENV=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))

def main():
    p=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
    o=p/'outputs';o.mkdir(exist_ok=True);(o/'context_crops').mkdir(exist_ok=True)
    sources=json.loads((p/'sources.json').read_text());jobs=[];pages=[];patches=[]
    start=time.monotonic()
    for source in sources:
        orig=Path(source['outputs']);spages=json.loads((orig/'pages.json').read_text())
        sjobs=json.loads((orig/'jobs.json').read_text());patches.extend(dict(q,source_run=source['id']) for q in json.loads(Path(source['patches']).read_text()))
        pages.extend(dict(q,source_run=source['id']) for q in spages)
        for pg in spages:
            image=cv2.imread(pg['path']);assert image is not None;h,w=image.shape[:2]
            for j in [j for j in sjobs if j['page_id']==pg['page_id']]:
                x,y,cw,ch=context_box(j['box'],w,h);crop=image[y:y+ch,x:x+cw]
                crop=cv2.resize(crop,None,fx=CONFIG['context_scale'],fy=CONFIG['context_scale'],interpolation=cv2.INTER_LANCZOS4)
                pad=CONFIG['context_padding'];crop=cv2.copyMakeBorder(crop,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(255,255,255))
                dest=o/'context_crops'/f"{len(jobs):04}.png";assert cv2.imwrite(str(dest),crop)
                jobs.append(dict(source_run=source['id'],page_id=j['page_id'],method=j['method'],index=j['index'],source_box=j['box'],context_box=[x,y,cw,ch],path=str(dest),context_width=crop.shape[1],context_height=crop.shape[0],pixels=crop.shape[0]*crop.shape[1]))
        print('prepared',source['id'],len(sjobs),'context jobs',flush=True)
    dump(o/'context_jobs.json',jobs)
    raw=o/'context_raw.jsonl';assert not raw.exists(),'refusing to overwrite context OCR'
    with raw.open('w') as f,(o/'native.log').open('w') as err:
        for offset in range(0,len(jobs),96):
            subprocess.run([str(p/'code/recognize'),*(j['path'] for j in jobs[offset:offset+96])],stdout=f,stderr=err,env=ENV,check=True)
            print('recognized',min(offset+96,len(jobs)),'/',len(jobs),flush=True)
    results=[json.loads(s) for s in raw.read_text().splitlines()];assert len(results)==len(jobs)
    index={}
    for j,r in zip(jobs,results,strict=True):
        assert j['path']==r['path'];index[(j['source_run'],j['page_id'],j['method'],j['index'])]=(j,r)
    decisions=[];accepted=[]
    for patch in patches:
        j,r=index[(patch['source_run'],patch['page_id'],patch['method'],patch['job_index'])]
        ok,reason,texts=localized_decision(patch,j,r)
        decision=dict(patch,accepted=ok,reject_reason=None if ok else reason,decision_reason=reason,context_texts=texts)
        decisions.append(decision)
        if ok:accepted.append(patch)
    dump(o/'pages.json',pages);dump(o/'decisions.json',decisions);dump(o/'patches.json',accepted)
    for pg in pages:
        for method in ('candidate','uniform'):
            text=[]
            for li,line in enumerate(pg['lines']):
                s=line['text']
                for a in sorted([a for a in accepted if a['page_id']==pg['page_id'] and a['method']==method and a['line_id']==li],key=lambda a:-a['offset']):
                    k=a['offset'];s=s[:k]+'<sup>'+a['identity']+'</sup>'+s[k:]
                text.append(s)
            dest=o/'repaired'/method;dest.mkdir(parents=True,exist_ok=True);(dest/(pg['page_id']+'.md')).write_text('\n'.join(text)+'\n')
    budget={}
    for method in ('candidate','uniform'):
        pairs=[(j,r) for j,r in zip(jobs,results) if j['method']==method]
        budget[method]=dict(context_calls=len(pairs),context_pixels=sum(j['pixels'] for j,r in pairs),ocr_ms=sum(r['elapsedMS'] for j,r in pairs))
    for pg in pages:
        aa=[j['pixels'] for j in jobs if j['page_id']==pg['page_id'] and j['method']=='candidate'];bb=[j['pixels'] for j in jobs if j['page_id']==pg['page_id'] and j['method']=='uniform'];assert aa==bb
    summary=dict(pages=len(pages),budget=budget,patches_before={m:sum(a['method']==m for a in patches) for m in budget},patches_after={m:sum(a['method']==m for a in accepted) for m in budget},rejections=dict(Counter(d['decision_reason'] for d in decisions if not d['accepted'])),wall_seconds=time.monotonic()-start,scope='Context gate reuses identical tight predictions. Extra context OCR for every paired crop; candidate/uniform exactly matched by page. Gate ablation is additional processing, not a no-extra-cost gain.')
    dump(o/'gate_summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
