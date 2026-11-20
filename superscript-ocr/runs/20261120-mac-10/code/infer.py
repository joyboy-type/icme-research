"""One frozen pixel-only replay. Reads no PDF text, JATS, gold or rids."""
from pathlib import Path
import json, re, subprocess, os, time, hashlib
from PIL import Image
from relations import citations, admit, words, read_bibliography

P=Path(__file__).resolve().parents[1];O=P/'outputs'
ENV=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def recognize(paths,output):
    assert not output.exists()
    with output.open('w') as out,(O/(output.stem+'.native.log')).open('w') as err:
        subprocess.run([str(P/'code/recognize'),'--chars',*map(str,paths)],stdout=out,stderr=err,check=True,env=ENV)
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows)==len(paths) and all(a['path']==str(b) for a,b in zip(rows,paths,strict=True))
    return rows

def located(raw,width,height,transform=lambda b:b):
    out=[]
    for li,line in enumerate(raw['lines']):
        if not line['candidates']:continue
        c=line['candidates'][0];text=c['text'];tokens=list(re.finditer(r'[0-9０-９⁰¹²³⁴⁵⁶⁷⁸⁹]+(?:[,，\-–−][0-9０-９⁰¹²³⁴⁵⁶⁷⁸⁹]+)*',text))
        for marker in citations(text):
            selected=[m for token,m in zip(tokens,c['matches']) if token.end()>marker['start'] and token.start()<marker['end']]
            if not selected:continue
            boxes=[[m['box'][0]*width,m['box'][1]*height,(m['box'][0]+m['box'][2])*width,(m['box'][1]+m['box'][3])*height] for m in selected]
            box=transform([min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)])
            out.append(dict(**marker,box=box,xy=[(box[0]+box[2])/2,(box[1]+box[3])/2],line_index=li,confidence=c['confidence']))
    return out

def main():
    assert not (O/'predictions.json').exists()
    start=time.monotonic();pages=json.loads((P/'inputs.json').read_text());protocol_sha=sha(P/'protocol.json')
    cache_replay=os.environ.get('RUN10_EXACT_RAW_REPLAY')=='1'
    whole=([json.loads(s) for s in (O/'whole_raw.jsonl').read_text().splitlines()] if cache_replay else recognize([p['path'] for p in pages],O/'whole_raw.jsonl'))
    books={};current={};started={}
    for page,raw in zip(pages,whole,strict=True):
        sid=page['source'];books.setdefault(sid,{});started.setdefault(sid,False)
        for line in raw['lines']:
            if not line['candidates']:continue
            text=line['candidates'][0]['text']
            if re.fullmatch(r'\s*References\s*',text,re.I):started[sid]=True;continue
            if not started[sid]:continue
            match=re.match(r'^\s*([1-9][0-9]{0,2})[.)]\s+(.+)',text)
            if match:
                n=int(match[1]);books[sid][n]=dict(text=match[2],page_id=page['id']);current[sid]=n
            elif sid in current:
                books[sid][current[sid]]['text']+=' '+text
    # Some Vision observations contain the reference number alone. Preserve that boundary.
    books={sid:read_bibliography([(page['id'],line['candidates'][0]['text']) for page,raw in zip(pages,whole,strict=True) if page['source']==sid for line in raw['lines'] if line['candidates']]) for sid in books}
    save(O/'ocr_bibliography.json',books)
    base=[];jobs=[];(O/'local_crops').mkdir(exist_ok=True)
    for page,raw in zip(pages,whole,strict=True):
        if not page['local']:continue
        found=located(raw,page['width'],page['height'])
        for f in found:base.append(dict(page_id=page['id'],source=page['source'],origin='whole',**f))
        centers=[dict(xy=f['xy'],left=f['left'],right=f['right'],kind='whole_bracket') for f in found[:4]]
        for line in raw['lines']:
            if not line['candidates']:continue
            c=line['candidates'][0]
            for m in c['matches']:
                xy=[m['x']*page['width'],m['y']*page['height']]
                if any(abs(xy[0]-q['xy'][0])<24 and abs(xy[1]-q['xy'][1])<24 for q in centers):continue
                centers.append(dict(xy=xy,left='',right='',kind='whole_numeric'))
                if len(centers)>=8:break
            if len(centers)>=8:break
        if len(centers)<8:
            for line in sorted(raw['lines'],key=lambda l:-len(l['candidates'][0]['text']) if l['candidates'] else 0):
                if len(centers)>=8:break
                x,y,w,h=line['box'];xy=[(x+w/2)*page['width'],(y+h/2)*page['height']]
                if any(abs(xy[0]-q['xy'][0])<24 and abs(xy[1]-q['xy'][1])<24 for q in centers):continue
                centers.append(dict(xy=xy,left='',right='',kind='long_line'))
        image=Image.open(page['path']).convert('L')
        for index in range(8):
            center=centers[index] if index<len(centers) else None;canvas=Image.new('L',(904,280),255)
            if center:
                cx,cy=center['xy'];x=max(0,min(page['width']-280,int(cx-140)));y=max(0,min(page['height']-72,int(cy-36)))
                crop=image.crop((x,y,x+280,y+72)).resize((840,216),Image.Resampling.LANCZOS);canvas.paste(crop,(32,32));box=[x,y,280,72]
            else:box=None
            fn=O/'local_crops'/f"{page['id']}-{index}.png";canvas.save(fn)
            jobs.append(dict(path=str(fn),page_id=page['id'],source=page['source'],index=index,source_box=box,center=center))
    save(O/'jobs.json',jobs)
    local=([json.loads(s) for s in (O/'local_raw.jsonl').read_text().splitlines()] if cache_replay else recognize([j['path'] for j in jobs],O/'local_raw.jsonl'));additions=[]
    assert len(whole)==len(pages) and len(local)==len(jobs)
    assert all(a['path']==j['path'] for a,j in zip(local,jobs,strict=True))
    raw_by_page={p['id']:r for p,r in zip(pages,whole)};byid={p['id']:p for p in pages}
    for ji,(job,raw) in enumerate(zip(jobs,local,strict=True)):
        if job['source_box'] is None:continue
        x,y,w,h=job['source_box'];transform=lambda b:[x+(b[0]-32)/3,y+(b[1]-32)/3,x+(b[2]-32)/3,y+(b[3]-32)/3]
        for candidate in located(raw,904,280,transform):
            if candidate['confidence']<.5:continue
            cx,cy=candidate['xy']
            if not x<=cx<=x+w or not y<=cy<=y+h:continue
            page=byid[job['page_id']];nearest=[]
            for line in raw_by_page[job['page_id']]['lines']:
                if not line['candidates']:continue
                a,b,c,d=line['box'];lx=a*page['width'];ly=b*page['height'];lw=c*page['width'];lh=d*page['height']
                if lx-15<=cx<=lx+lw+15 and abs(cy-(ly+lh/2))<=max(18,lh):
                    nearest.append((abs(cy-(ly+lh/2)),line))
            if not nearest:before=after=''
            else:
                line=min(nearest,key=lambda z:z[0])[1];a,b,c,d=line['box'];fraction=max(0,min(1,(cx-a*page['width'])/(c*page['width'])))
                text=line['candidates'][0]['text'];k=round(len(text)*fraction)
                # Anchor around the nearest whole-OCR bracket when one exists.
                matches=citations(text)
                if matches:k=min(matches,key=lambda z:abs(z['start']-k))['start'];end=next(z['end'] for z in matches if z['start']==k)
                else:end=k
                before=text[:k];after=text[end:]
            allowed=admit(candidate,books[job['source']],before,after,True)
            additions.append(dict(page_id=job['page_id'],source=job['source'],origin='local',job_index=ji,anchor_before=before,anchor_after=after,consistent=allowed,**candidate))
    methods={}
    for method in ['whole','local_mapping','local_consistency']:
        rows=[dict(b) for b in base]
        if method!='whole':
            for candidate in additions:
                if method=='local_consistency' and not candidate['consistent']:continue
                rows=[r for r in rows if not(r['page_id']==candidate['page_id'] and abs(r['xy'][0]-candidate['xy'][0])<24 and abs(r['xy'][1]-candidate['xy'][1])<18)]
                rows.append(dict(candidate))
        for row in rows:
            row['targets']=[books[row['source']].get(n) for n in row['numbers']]
            row['auto_insert']=False;row['review_required']=True
        methods[method]=rows
    save(O/'local_candidates.json',additions);save(O/'predictions.json',methods)
    assert sha(P/'protocol.json')==protocol_sha
    meta=dict(id=P.name,status='inference_complete',automatic_insertion=False,protocol_sha256=protocol_sha,whole_calls=len(whole),local_calls=len(local),whole_pixels=sum(p['width']*p['height'] for p in pages),local_pixels=len(jobs)*904*280,shared_local_evidence=True,bibliography_entries={k:len(v) for k,v in books.items()},wall_seconds=time.monotonic()-start,model_downloads=0,training=False,paid_calls=0,code_sha256={f.name:sha(f) for f in (P/'code').glob('*') if f.suffix in ['.py','.swift']})
    meta.update(exact_raw_replay=cache_replay,additional_ocr_calls_on_replay=0 if cache_replay else None)
    save(P/'run.json',meta);print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
