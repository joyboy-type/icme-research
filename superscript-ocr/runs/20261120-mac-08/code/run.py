"""Matched-canvas development ablation; no labels are read during inference."""
from pathlib import Path
import json,subprocess,os,hashlib,math,time
import cv2
import numpy as np
import baseline
from candidates import CONFIG,rerank,render

P=Path(__file__).resolve().parents[1];O=P/'outputs'
METHODS=['baseline','rerank','isolate','uniform']
ENV=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    sources=json.loads((P/'sources.json').read_text());pages=[]
    for source in sources:
        pages.extend(dict(pg,source_run=source['id']) for pg in json.loads(Path(source['pages']).read_text()))
    assert len(pages)==24 and len({p['page_id'] for p in pages})==24
    taskbook=Path(json.loads((P/'protocol.json').read_text())['frozen_taskbook']);frozen_hash=sha(taskbook)
    meta=dict(id=P.name,status='running',type='development reranking and component-isolation ablation',config=CONFIG,
              methods=METHODS,pages=24,uses_gold_boxes=False,paid_calls=0,training=False,model_downloads=0,
              gold_use='Only separate score.py after raw outputs saved; no labels read by inference.',
              input_page_hashes=[dict(page_id=p['page_id'],sha256=sha(Path(p['path']))) for p in pages],
              code_sha256={f.name:sha(f) for f in (P/'code').glob('*') if f.suffix in ['.py','.swift']},
              frozen_taskbook_sha256=frozen_hash,execution='local serial Vision requests CPU-only flag; OpenCV/BLAS2threads; Vision internal scheduling not exposed',
              budget_policy='Exactly32calls per page per method, per-page common white canvas, common4x source scale. Missing proposals occupy blank slots. True source pixel counts separately reported.',
              common_preprocessing='Same cached full-page OCR, grayscale source and adaptive-threshold components for all methods; fresh baseline uses common canvases and is not numerically interchangeable with archived old baseline.')
    save(P/'run.json',meta);start=time.monotonic();jobs=[];(O/'crops').mkdir(exist_ok=True)
    for pi,page in enumerate(pages):
        gray=cv2.cvtColor(cv2.imread(page['path']),cv2.COLOR_BGR2GRAY);ih,iw=gray.shape
        simple=[dict(text=l['text'],box=(np.array(l['box'])/[iw,ih,iw,ih]).tolist()) for l in page['lines']]
        baseline.CONFIG['max_crops']=32;old=baseline.proposals(gray,simple)
        assert [p['box'] for p in old]==[p['box'] for p in page['proposals']],'old proposal replay differs'
        baseline.CONFIG['max_crops']=100000;pool=baseline.proposals(gray,simple);baseline.CONFIG['max_crops']=32
        mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
        _,cc,stats,_=cv2.connectedComponentsWithStats(mask)
        new=rerank(pool,cc,stats,32);page['new_proposals']=new;page['untruncated_old_candidates']=len(pool)
        allboxes=[p['box'] for p in old]+[p['box'] for p in new]+[p['raw_box'] for p in new]
        cw=max(256,math.ceil((max((b[2]-b[0] for b in allboxes),default=20)*4+64)/16)*16)
        ch=max(128,math.ceil((max((b[3]-b[1] for b in allboxes),default=20)*4+64)/16)*16)
        page['canvas']=[cw,ch]
        for index in range(32):
            for method in METHODS:
                proposal=old[index] if method=='baseline' and index<len(old) else new[index] if method in ['rerank','isolate'] and index<len(new) else None
                box=None;image=None
                if method=='uniform':
                    reference=new[index]['box'] if index<len(new) else [0,0,20,20]
                    w,h=reference[2]-reference[0],reference[3]-reference[1]
                    x=round(baseline.halton(index+1,2)*(iw-w));y=round(baseline.halton(index+1,3)*(ih-h));box=[x,y,x+w,y+h]
                    image=gray[y:y+h,x:x+w]
                elif proposal is not None:
                    box=proposal['raw_box'] if method=='rerank' else proposal['box'];l,t,r,b=box
                    image=render(gray,cc,proposal) if method=='isolate' else gray[t:b,l:r]
                canvas=np.full((ch,cw),255,np.uint8);padding=[0,0];source_pixels=0
                if image is not None:
                    source_pixels=image.shape[0]*image.shape[1];resized=cv2.resize(image,None,fx=4,fy=4,interpolation=cv2.INTER_LANCZOS4);hh,ww=resized.shape
                    px,py=(cw-ww)//2,(ch-hh)//2;assert px>=32 and py>=32
                    canvas[py:py+hh,px:px+ww]=resized;padding=[px,py]
                path=O/'crops'/f'{pi:02}_{method}_{index:02}.png';assert cv2.imwrite(str(path),canvas)
                jobs.append(dict(page_id=page['page_id'],method=method,index=index,path=str(path),source_box=box,scale=4,padding=padding,width=cw,height=ch,input_pixels=cw*ch,source_pixels=source_pixels,blank=box is None))
        print('prepared',pi+1,'old',len(old),'new',len(new),'canvas',cw,ch,flush=True)
    save(O/'pages.json',pages);save(O/'jobs.json',jobs);assert len(jobs)==3072
    raw=O/'raw.jsonl';assert not raw.exists()
    with raw.open('w') as out,(O/'native.log').open('w') as err:
        for offset in range(0,len(jobs),96):
            subprocess.run([str(P/'code/recognize'),*(j['path'] for j in jobs[offset:offset+96])],stdout=out,stderr=err,env=ENV,check=True)
            print('recognized',min(offset+96,len(jobs)),'/',len(jobs),flush=True)
    outputs=[json.loads(s) for s in raw.read_text().splitlines()];assert len(outputs)==len(jobs)
    predictions=[]
    for job,result in zip(jobs,outputs,strict=True):
        assert job['path']==result['path'];job['raw_top1']=' | '.join(l['candidates'][0]['text'] for l in result['lines'] if l['candidates'])
        if job['blank']:continue
        x0,y0,x1,y1=job['source_box'];padx,pady=job['padding']
        for line in result['lines']:
            if not line['candidates']:continue
            c=line['candidates'][0]
            if c['confidence']<.5:continue
            for m in c['matches']:
                x,y,w,h=m['box'];l=x0+(x*job['width']-padx)/4;t=y0+(y*job['height']-pady)/4;r=l+w*job['width']/4;b=t+h*job['height']/4;cx,cy=(l+r)/2,(t+b)/2
                if not x0<=cx<=x1 or not y0<=cy<=y1:continue
                predictions.append(dict(page_id=job['page_id'],method=job['method'],job_index=job['index'],text=baseline.numeric(m['text']),xy=[cx,cy],box=[l,t,r,b],confidence=c['confidence']))
    patches=[]
    for page in pages:
        for method in METHODS:
            pp=[p for p in predictions if p['page_id']==page['page_id'] and p['method']==method]
            added=baseline.attach(page['lines'],pp,page['numbers']);patches.extend(added)
            lines=[]
            for li,line in enumerate(page['lines']):
                text=line['text']
                for p in sorted([a for a in added if a['line_id']==li],key=lambda a:-a['offset']):
                    k=p['offset'];text=text[:k]+'<sup>'+p['identity']+'</sup>'+text[k:]
                lines.append(text)
            folder=O/'repaired'/method;folder.mkdir(parents=True,exist_ok=True);(folder/(page['page_id']+'.md')).write_text('\n'.join(lines)+'\n')
    save(O/'jobs.json',jobs);save(O/'predictions.json',predictions);save(O/'patches.json',patches)
    budgets={m:dict(calls=sum(j['method']==m for j in jobs),input_pixels=sum(j['input_pixels'] for j in jobs if j['method']==m),source_pixels=sum(j['source_pixels'] for j in jobs if j['method']==m),blank_slots=sum(j['method']==m and j['blank'] for j in jobs),ocr_ms=sum(r['elapsedMS'] for j,r in zip(jobs,outputs) if j['method']==m)) for m in METHODS}
    assert len({(b['calls'],b['input_pixels']) for b in budgets.values()})==1
    assert sha(taskbook)==frozen_hash
    meta.update(status='inference_complete',budgets=budgets,patches={m:sum(a['method']==m for a in patches) for m in METHODS},wall_seconds=time.monotonic()-start,taskbook_unchanged=True,common_whole_page_ocr_reused_calls=24,new_whole_page_calls=0);save(P/'run.json',meta);print(json.dumps({k:meta[k] for k in ['budgets','patches','wall_seconds']},indent=2))
if __name__=='__main__':main()
