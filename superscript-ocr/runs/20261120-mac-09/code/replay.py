"""One unchanged frozen gate, exact visual-context cache reuse, no threshold search."""
from pathlib import Path
import json,hashlib,subprocess,os,time
import cv2,numpy as np
from gate import context_box,localized_decision,CONFIG

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def evidence_key(image_sha,box,engine_sha):
    payload=dict(source_image=image_sha,context_box=box,engine=engine_sha,
                 scale=CONFIG['context_scale'],padding=CONFIG['context_padding'])
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    p=Path(__file__).resolve().parents[1];r=p.parent;o=p/'outputs';base=r/'20261120-mac-08';cv2.setNumThreads(2)
    assert not (o/'decisions.json').exists(),'refusing to overwrite'
    protocol=json.loads((p/'protocol.json').read_text());engine=sha(p/'code/recognize.swift')
    assert engine==protocol['engine_sha256'] and sha(p/'code/gate.py')==protocol['frozen_gate_sha256']
    frozen_word=Path(json.loads((base/'protocol.json').read_text())['frozen_taskbook']);assert sha(frozen_word)==protocol['taskbook_frozen_sha256']
    started=time.monotonic();pages=json.loads((base/'outputs/pages.json').read_text());byid={pg['page_id']:pg for pg in pages}
    image_sha={pg['page_id']:sha(Path(pg['path'])) for pg in pages}
    rawpatches=[a for a in json.loads((base/'outputs/patches.json').read_text()) if a['method'] in ['baseline','isolate']]
    oldjobs=json.loads((base/'outputs/jobs.json').read_text());oldindex={(j['page_id'],j['method'],j['index']):j for j in oldjobs}
    cache={};cache_duplicates=0
    for rid in ['20261120-mac-05','20261120-mac-06']:
        q=r/rid;assert sha(q/'code/recognize.swift')==engine
        js=json.loads((q/'outputs/context_jobs.json').read_text());rs=[json.loads(s) for s in (q/'outputs/context_raw.jsonl').read_text().splitlines()]
        for index,(j,out) in enumerate(zip(js,rs,strict=True)):
            assert j['path']==out['path'];pid=j['page_id']
            if pid not in image_sha:continue
            key=evidence_key(image_sha[pid],j['context_box'],engine)
            if key in cache:cache_duplicates+=1
            else:cache[key]=dict(job=j,result=out,source_run=rid,source_index=index)
    evidence={};cases=[];images={};fresh=[];(o/'new_contexts').mkdir(exist_ok=True)
    for index,patch in enumerate(rawpatches):
        pid=patch['page_id'];pg=byid[pid];job=oldindex[pid,patch['method'],patch['job_index']]
        parent=job['source_box'] if patch['method']=='baseline' else pg['new_proposals'][patch['job_index']]['raw_box']
        # The parent proposal is the common unmasked visual frame for raw and isolated OCR.
        x,y,w,h=context_box(parent,pg['width'],pg['height']);box=[x,y,w,h];key=evidence_key(image_sha[pid],box,engine)
        cases.append(dict(index=index,patch=patch,key=key,parent_box=parent))
        if key in evidence:continue
        if pid not in images:images[pid]=cv2.imread(pg['path'])
        crop=images[pid][y:y+h,x:x+w];crop=cv2.resize(crop,None,fx=CONFIG['context_scale'],fy=CONFIG['context_scale'],interpolation=cv2.INTER_LANCZOS4);pad=CONFIG['context_padding'];crop=cv2.copyMakeBorder(crop,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(255,255,255))
        ejob=dict(context_box=box,context_width=crop.shape[1],context_height=crop.shape[0],pixels=crop.shape[0]*crop.shape[1])
        if key in cache:
            prior=cache[key];actual=cv2.imread(prior['job']['path']);assert actual is not None and np.array_equal(crop,actual),'cache pixels differ'
            evidence[key]=dict(key=key,page_id=pid,job=ejob,result=prior['result'],origin='cache',source_run=prior['source_run'],source_index=prior['source_index'],source_png_sha256=sha(Path(prior['job']['path'])),pixels_verified=True)
        else:
            fn=o/'new_contexts'/(key+'.png');assert cv2.imwrite(str(fn),crop)
            evidence[key]=dict(key=key,page_id=pid,job=ejob,origin='new',path=str(fn));fresh.append(key)
    print('cases',len(cases),'unique evidence',len(evidence),'cached',sum(v['origin']=='cache' for v in evidence.values()),'new',len(fresh),flush=True)
    save(o/'case_inputs.json',cases)
    raw=o/'new_raw.jsonl';assert not raw.exists()
    if fresh:
        with raw.open('w') as out,(o/'native.log').open('w') as err:
            subprocess.run([str(p/'code/recognize'),*(evidence[k]['path'] for k in fresh)],stdout=out,stderr=err,check=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2'))
        results=[json.loads(s) for s in raw.read_text().splitlines()];assert len(results)==len(fresh)
        for key,result in zip(fresh,results,strict=True):assert evidence[key]['path']==result['path'];evidence[key]['result']=result
    else:raw.write_text('')
    decisions=[]
    for case in cases:
        ev=evidence[case['key']];patch=case['patch'];ok,reason,texts=localized_decision(patch,ev['job'],ev['result'])
        decisions.append(dict(index=case['index'],**patch,evidence_key=case['key'],evidence_origin=ev['origin'],accepted=ok,decision_reason=reason,context_texts=texts))
    save(o/'decisions.json',decisions);save(o/'evidence.json',list(evidence.values()))
    # This run intentionally emits a review queue, never an automatically repaired transcript.
    save(o/'review_candidates.json',[dict(d,review_required=True,auto_insert=False) for d in decisions])
    meta=dict(id=p.name,status='inference_complete',type='frozen gate development audit on run08 candidates',paid_calls=0,new_model_downloads=0,training=False,new_source_pages=0,
              gate_sha256=sha(p/'code/gate.py'),engine_sha256=engine,frozen_word_sha256=sha(frozen_word),word_unchanged=sha(frozen_word)==protocol['taskbook_frozen_sha256'],
              candidate_cases=len(cases),unique_visual_evidence=len(evidence),cached_evidence=sum(e['origin']=='cache' for e in evidence.values()),new_local_ocr_calls=len(fresh),
              fresh_context_pixels=sum(evidence[k]['job']['pixels'] for k in fresh),cache_duplicate_keys=cache_duplicates,wall_seconds=time.monotonic()-started,
              execution='local serial Vision accurate revision3, CPU-only request flag; OpenCV/BLAS limits2; exact frozen gate unchanged',
              inputs={'run08_patches_sha256':sha(base/'outputs/patches.json'),'run08_jobs_sha256':sha(base/'outputs/jobs.json')},
              budget_note='Cache-first semantic replay only on emitted additions, not a new equal-cost ranking benchmark. Baseline/isolate evidence counts differ and all new calls/pixels are reported.',
              no_threshold_search=True,automatic_insertion=False,output_policy='Review queue only pending acceptance check; no repaired transcript emitted.')
    assert meta['word_unchanged'];save(p/'run.json',meta)
    print(json.dumps({k:meta[k] for k in ['candidate_cases','unique_visual_evidence','cached_evidence','new_local_ocr_calls','word_unchanged']},indent=2))
if __name__=='__main__':main()
