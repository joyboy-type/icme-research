from pathlib import Path
import json, subprocess, os, hashlib, sys, time, re
from datetime import datetime, timezone
import cv2
import numpy as np
from method import proposals, paired_jobs, attach, numeric, CONFIG

P=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
O=P/'outputs'; O.mkdir(exist_ok=True); (O/'crops').mkdir(exist_ok=True)
ENV=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',MKL_NUM_THREADS='2')
def dump(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def ocr(paths,name,chars=False):
    dest=O/(name+'.jsonl'); assert not dest.exists(), f'refusing to overwrite {dest}'
    with dest.open('w') as f,(O/(name+'.log')).open('w') as err:
        subprocess.run([str(P/'code/recognize')]+(['--chars'] if chars else [])+paths,stdout=f,stderr=err,env=ENV,check=True)
    out=[json.loads(s) for s in dest.read_text().splitlines()]
    assert len(out)==len(paths)
    return out

def main():
    pages=json.loads((P/'pages.json').read_text())
    meta=dict(id=P.name,status='running',started=datetime.now(timezone.utc).isoformat(),
              config=CONFIG,charged_calls=0,model_downloads=0,training=False,
              execution='local Mac; serial Vision requests cpuOnly=true; OpenCV and BLAS limits2; Vision internal scheduling not exposed',
              method_files={p.name:sha(p) for p in (P/'code').glob('*') if p.suffix in ('.py','.swift') and p.name!='test_method.py'},
              input_images=[dict(path=p['path'],sha256=sha(Path(p['path']))) for p in pages],
              gold_use='No labels in proposal, OCR, acceptance or text insertion. Separate score.py after raw outputs saved.')
    dump(P/'run.json',meta)
    started=time.monotonic()
    full=ocr([p['path'] for p in pages],'full',True)
    jobs=[]; normalized=[]
    for pi,(page,result) in enumerate(zip(pages,full,strict=True)):
        image=cv2.imread(page['path']); h,w=image.shape[:2]
        lines=[]; nums=[]
        for line in result['lines']:
            if not line['candidates']:continue
            top=line['candidates'][0]; lb=(np.array(line['box'])*[w,h,w,h]).tolist()
            chars=[dict(c,box=(np.array(c['box'])*[w,h,w,h]).tolist()) for c in top['chars']]
            lines.append(dict(text=top['text'],box=lb,chars=chars))
            for m in top['matches']:
                bb=(np.array(m['box'])*[w,h,w,h]).tolist(); x,y,bw,bh=bb
                nums.append(dict(text=numeric(m['text']),xy=[m['x']*w,m['y']*h],box=[x,y,x+bw,y+bh]))
        simple=[dict(text=l['text'],box=(np.array(l['box'])/[w,h,w,h]).tolist()) for l in lines]
        props=proposals(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),simple)
        for j in paired_jobs(props,w,h):
            x1,y1,x2,y2=j['box']; crop=image[y1:y2,x1:x2]
            crop=cv2.resize(crop,None,fx=CONFIG['scale'],fy=CONFIG['scale'],interpolation=cv2.INTER_LANCZOS4)
            pad=CONFIG['padding'];crop=cv2.copyMakeBorder(crop,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(255,255,255))
            fn=O/'crops'/f"{pi:02}_{j['method']}_{j['index']:02}.png";assert cv2.imwrite(str(fn),crop)
            jobs.append(dict(j,page_id=page['page_id'],path=str(fn),width=crop.shape[1],height=crop.shape[0],pixels=crop.shape[0]*crop.shape[1]))
        normalized.append(dict(page_id=page['page_id'],path=page['path'],width=w,height=h,lines=lines,numbers=nums,proposals=props))
        print('prepared',page['page_id'],len(props),'proposals',flush=True)
    dump(O/'pages.json',normalized);dump(O/'jobs.json',jobs)
    crops=ocr([j['path'] for j in jobs],'crops')
    pred=[]
    for j,result in zip(jobs,crops,strict=True):
        texts=[]
        for line in result['lines']:
            if not line['candidates']:continue
            top=line['candidates'][0];texts.append(top['text'])
            if top['confidence'] < CONFIG['confidence']:continue
            for m in top['matches']:
                x,y,bw,bh=m['box']; scale=CONFIG['scale'];pad=CONFIG['padding'];x0,y0,_,_=j['box']
                l=x0+(x*j['width']-pad)/scale;t=y0+(y*j['height']-pad)/scale
                r=l+bw*j['width']/scale;b=t+bh*j['height']/scale
                # Ignore observations centered in padding rather than source pixels.
                cx,cy=(l+r)/2,(t+b)/2
                if not (j['box'][0]<=cx<=j['box'][2] and j['box'][1]<=cy<=j['box'][3]):continue
                pred.append(dict(page_id=j['page_id'],method=j['method'],job_index=j['index'],text=numeric(m['text']),xy=[cx,cy],box=[l,t,r,b],confidence=top['confidence']))
        j['raw_top1']=' | '.join(texts)
    dump(O/'jobs.json',jobs);dump(O/'predictions.json',pred)
    patches=[]
    for page in normalized:
        for method in ('candidate','uniform'):
            pp=[p for p in pred if p['page_id']==page['page_id'] and p['method']==method]
            current=attach(page['lines'],pp,page['numbers']);patches.extend(current)
            text=[]
            for i,line in enumerate(page['lines']):
                s=line['text']
                for patch in sorted([p for p in current if p['line_id']==i],key=lambda p:-p['offset']):
                    k=patch['offset'];s=s[:k]+f"<sup>{patch['identity']}</sup>"+s[k:]
                text.append(s)
            folder=O/'repaired'/method;folder.mkdir(parents=True,exist_ok=True)
            (folder/(page['page_id']+'.md')).write_text('\n'.join(text)+'\n')
    dump(O/'patches.json',patches)
    budget={}
    for method in ('candidate','uniform'):
        js=[j for j in jobs if j['method']==method]
        budget[method]=dict(calls=len(js),input_pixels=sum(j['pixels'] for j in js),ocr_ms=sum(r['elapsedMS'] for j,r in zip(jobs,crops) if j['method']==method))
    assert budget['candidate']['calls']==budget['uniform']['calls']
    assert budget['candidate']['input_pixels']==budget['uniform']['input_pixels']
    for page in normalized:
        a=[j['pixels'] for j in jobs if j['page_id']==page['page_id'] and j['method']=='candidate']
        b=[j['pixels'] for j in jobs if j['page_id']==page['page_id'] and j['method']=='uniform'];assert a==b
    meta.update(status='inference_complete',wall_seconds=time.monotonic()-started,budget=budget,
                common_fullpage=dict(calls=len(full),input_pixels=sum(p['width']*p['height'] for p in normalized),ocr_ms=sum(r['elapsedMS'] for r in full)),
                patches={m:sum(p['method']==m for p in patches) for m in budget})
    dump(P/'run.json',meta);print(json.dumps({k:meta[k] for k in ('budget','patches','wall_seconds')},indent=2))
if __name__=='__main__':main()
