"""Oracle recognition and ink-coverage diagnosis on already inspected run06 pages.

This is a gold-box diagnostic, never an inference policy. It never edits a taskbook.
"""
from pathlib import Path
import json,csv,math,subprocess,os,sys,hashlib,unicodedata,importlib.util
from collections import Counter
import cv2
import numpy as np

def ink_coverage(points,box):
    l,t,r,b=box
    return sum(l<=x<r and t<=y<b for x,y in points)/len(points) if points else 0.

def numeric(s):return unicodedata.normalize('NFKC',str(s)).translate(str.maketrans('–−—','---')).strip().strip('[]')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2))

def main():
    p=Path(__file__).resolve().parents[1];o=p/'outputs';old=p.parent/'20261120-mac-06'
    protocol=Path(sys.argv[1]) if len(sys.argv)>1 else None
    frozen_protocol_hash=sha(protocol) if protocol else None
    labels=json.loads((old/'base/outputs/evaluation_labels.json').read_text());assert len(labels)==17
    pages=json.loads((old/'base/outputs/pages.json').read_text());cv2.setNumThreads(2)
    methods=['tight_native','tight_4x'];jobs=[];images={pg['page_id']:cv2.imread(pg['path']) for pg in pages}
    config=dict(conditions=methods,margin_source_pixels=2,scale={'tight_native':1,'tight_4x':4},padding={'tight_native':16,'tight_4x':32},minimum_canvas=64,ink_threshold=180,complete_ink_fraction=.98,score='numeric_identity_v2; first candidate primary, top3 inclusion diagnostic only',scope='17 previously inspected references on6Chinese report pages; oracle boxes only; not a new confirmation set or deployable performance.')
    meta=dict(id=p.name,status='running',type='oracle recognition and stage decomposition',config=config,charged_calls=0,model_downloads=0,training=False,uses_gold_boxes=True,taskbook_frozen_sha256=frozen_protocol_hash,execution='local serial Vision accurate revision3; CPU-only request flag; OpenCV and BLAS limits2; Vision internal scheduling not exposed',inputs=[dict(kind='labels',sha256=sha(old/'base/outputs/evaluation_labels.json')),dict(kind='old_jobs',sha256=sha(old/'base/outputs/jobs.json'))]+[dict(page_id=pg['page_id'],sha256=sha(Path(pg['path']))) for pg in pages],code_sha256={n:sha(p/'code'/n) for n in ['recognize.swift','diagnose.py']})
    save(p/'run.json',meta);save(p/'config.json',config)
    for m in labels:
        im=images[m['page_id']];ih,iw=im.shape[:2];poly=m['span']['poly'];l,t,r,b=min(poly[::2]),min(poly[1::2]),max(poly[::2]),max(poly[1::2])
        x1,y1,x2,y2=max(0,math.floor(l)-2),max(0,math.floor(t)-2),min(iw,math.ceil(r)+2),min(ih,math.ceil(b)+2)
        for method in methods:
            scale=config['scale'][method];crop=im[y1:y2,x1:x2]
            if scale>1:crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_LANCZOS4)
            pad=config['padding'][method];hh,ww=crop.shape[:2];px=max(pad,math.ceil((64-ww)/2));py=max(pad,math.ceil((64-hh)/2));crop=cv2.copyMakeBorder(crop,py,py,px,px,cv2.BORDER_CONSTANT,value=(255,255,255));fn=o/'crops'/f"{m['marker_id']}_{method}.png";assert cv2.imwrite(str(fn),crop)
            jobs.append(dict(marker_id=m['marker_id'],page_id=m['page_id'],method=method,path=str(fn),source_box=[x1,y1,x2,y2],gold_box=[l,t,r,b],scale=scale,padding=[px,py],width=crop.shape[1],height=crop.shape[0],pixels=crop.shape[0]*crop.shape[1]))
    save(o/'jobs.json',jobs);raw=o/'raw.jsonl';assert not raw.exists(),'refusing to overwrite'
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
    with raw.open('w') as out,(o/'native.log').open('w') as err:subprocess.run([str(p/'code/recognize'),*(j['path'] for j in jobs)],stdout=out,stderr=err,check=True,env=env)
    preds=[json.loads(s) for s in raw.read_text().splitlines()];assert len(preds)==34
    byid={m['marker_id']:m for m in labels};rows=[]
    for j,pred in zip(jobs,preds,strict=True):
        assert j['path']==pred['path'];m=byid[j['marker_id']];l,t,r,b=j['gold_box'];mx,my=max(5,(r-l)*.2),max(5,(b-t)*.2);tokens=[]
        for line in pred['lines']:
            for c in line['candidates']:
                for token in c['matches']:
                    x=j['source_box'][0]+(token['x']*j['width']-j['padding'][0])/j['scale'];y=j['source_box'][1]+(token['y']*j['height']-j['padding'][1])/j['scale']
                    tokens.append(dict(rank=c['rank'],text=numeric(token['text']),xy=[x,y],at_gold=l-mx<=x<=r+mx and t-my<=y<=b+my))
        exact=lambda q:q['text']==numeric(m['identity'])
        rows.append(dict(marker_id=m['marker_id'],page_id=m['page_id'],method=j['method'],gold_identity=m['identity'],top1_exact=any(q['rank']==1 and exact(q) for q in tokens),top1_exact_at_box=any(q['rank']==1 and exact(q) and q['at_gold'] for q in tokens),top3_contains_exact_at_box=any(exact(q) and q['at_gold'] for q in tokens),observation_count=len(pred['lines']),raw_top1=' | '.join(line['candidates'][0]['text'] for line in pred['lines'] if line['candidates']),tokens=tokens,pixels=j['pixels'],elapsed_ms=pred['elapsedMS']))
    save(o/'oracle_rows.json',rows)
    # Re-enumerate the original method without the top32 truncation only for diagnosis.
    module_path=old/'base/code/method.py';spec=importlib.util.spec_from_file_location('previous_method',module_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    geometry={};ranked={}
    oldjobs=json.loads((old/'base/outputs/jobs.json').read_text());oldrows=json.loads((old/'base/outputs/per_marker.json').read_text());oldrow={r['marker_id']:r for r in oldrows}
    for pg in pages:
        im=images[pg['page_id']];ih,iw=im.shape[:2];simple=[dict(text=l['text'],box=(np.array(l['box'])/[iw,ih,iw,ih]).tolist()) for l in pg['lines']]
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);default=module.proposals(gray,simple)
        assert [q['box'] for q in default]==[q['box'] for q in pg['proposals']],'old proposal replay changed'
        module.CONFIG['max_crops']=100000;allp=module.proposals(gray,simple);module.CONFIG['max_crops']=32;ranked[pg['page_id']]=allp
    stages=[]
    for m in labels:
        im=images[m['page_id']];gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);poly=m['span']['poly'];l,t,r,b=min(poly[::2]),min(poly[1::2]),max(poly[::2]),max(poly[1::2]);x0,y0,x1,y1=math.floor(l),math.floor(t),math.ceil(r),math.ceil(b)
        yy,xx=np.where(gray[y0:y1,x0:x1]<config['ink_threshold']);points=list(zip((xx+x0).tolist(),(yy+y0).tolist()));assert points,'empty reference ink'
        actual=[j for j in oldjobs if j['page_id']==m['page_id'] and j['method']=='candidate'];coverage=[ink_coverage(points,j['box']) for j in actual];best=max(range(len(actual)),key=lambda i:coverage[i]) if actual else None;fraction=coverage[best] if best is not None else 0.
        allp=ranked[m['page_id']];allcoverage=[ink_coverage(points,q['box']) for q in allp];complete_ranks=[i+1 for i,cov in enumerate(allcoverage) if cov>=config['complete_ink_fraction']]
        complete=fraction>=config['complete_ink_fraction'];part=fraction>0
        stage='no_reference_ink_in_selected_crops' if not part else 'partial_reference_ink_only' if not complete else 'complete_crop_but_number_not_read' if not oldrow[m['marker_id']]['candidate_hit'] else 'read_but_not_inserted'
        oracle=next(q for q in rows if q['marker_id']==m['marker_id'] and q['method']=='tight_4x')
        stages.append(dict(marker_id=m['marker_id'],page_id=m['page_id'],gold_identity=m['identity'],stage=stage,selected_crop_center_covered=oldrow[m['marker_id']]['candidate_covered'],best_selected_ink_fraction=fraction,best_selected_box=actual[best]['box'] if best is not None else None,complete_selected_crop=complete,first_complete_rank_before_budget=min(complete_ranks) if complete_ranks else None,all_proposals_before_budget=len(allp),old_candidate_exact_at_position=oldrow[m['marker_id']]['candidate_hit'],old_inserted_at_anchor=oldrow[m['marker_id']]['candidate_inserted_at_anchor'],oracle4x_exact_at_box=oracle['top1_exact_at_box'],oracle4x_raw=oracle['raw_top1']))
    save(o/'stage_rows.json',stages)
    with (o/'stage_rows.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(stages[0]));w.writeheader();w.writerows(stages)
    summary=dict(markers=17,pages=6,methods={method:{'calls':17,'top1_exact':sum(q['top1_exact'] for q in rows if q['method']==method),'top1_exact_at_box':sum(q['top1_exact_at_box'] for q in rows if q['method']==method),'top3_contains_exact_at_box':sum(q['top3_contains_exact_at_box'] for q in rows if q['method']==method),'pixels':sum(q['pixels'] for q in rows if q['method']==method),'elapsed_ms':sum(q['elapsed_ms'] for q in rows if q['method']==method)} for method in methods},stage_counts=dict(Counter(s['stage'] for s in stages)),complete_crop_present_before_top32=sum(s['first_complete_rank_before_budget'] is not None for s in stages),complete_crop_excluded_by_top32=sum(s['first_complete_rank_before_budget'] is not None and s['first_complete_rank_before_budget']>32 for s in stages),scope=config['scope'],coverage_note='Coverage uses source pixels gray<180 within annotated box; complete means>=98% of that ink in one crop. This is an evaluation proxy requiring visual review, not an inference feature.',taskbook_unchanged=(sha(protocol)==frozen_protocol_hash) if protocol else None)
    assert summary['taskbook_unchanged'] is not False
    save(o/'summary.json',summary);meta.update(status='complete',outcome=summary,taskbook_unchanged=summary['taskbook_unchanged']);save(p/'run.json',meta);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
