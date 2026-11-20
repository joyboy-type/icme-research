"""Post-inspection diagnostic amendment: remove punctuation within annotation boxes.

This condition is post hoc and gold-guided; it is not a deployable method.
"""
from pathlib import Path
import json,cv2,math,subprocess,os,importlib.util
import numpy as np
from diagnose import numeric,ink_coverage,save
p=Path(__file__).resolve().parents[1];o=p/'outputs';old=p.parent/'20261120-mac-06';cv2.setNumThreads(2)
labels=json.loads((old/'base/outputs/evaluation_labels.json').read_text());pages=json.loads((old/'base/outputs/pages.json').read_text());jobs=[];pointmap={}
save(o/'glyph_amendment.json',dict(reason='Visual audit found sentence punctuation and neighboring glyph fragments inside/adjacent to annotated boxes; isolate only top-aligned numeral components to separate crop contamination from recognition.',selection='gray<180 components within annotation, area>=3px; keep components whose top<=minimum component top+2px, visually inspect every final crop; zero source margin, white padding32, scale4',post_hoc=True,uses_gold=True))
for m in labels:
 pg=next(q for q in pages if q['page_id']==m['page_id']);im=cv2.imread(pg['path']);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);poly=m['span']['poly'];l,t,r,b=math.floor(min(poly[::2])),math.floor(min(poly[1::2])),math.ceil(max(poly[::2])),math.ceil(max(poly[1::2]));mask=(gray[t:b,l:r]<180).astype('uint8');_,cc,stats,_=cv2.connectedComponentsWithStats(mask);components=[(i,row) for i,row in enumerate(stats) if i and row[4]>=3];top=min(row[1] for i,row in components);selected=[(i,row) for i,row in components if row[1]<=top+2];sx=min(row[0] for i,row in selected);sy=min(row[1] for i,row in selected);ex=max(row[0]+row[2] for i,row in selected);ey=max(row[1]+row[3] for i,row in selected);box=[int(l+sx),int(t+sy),int(l+ex),int(t+ey)];yy,xx=np.where(np.isin(cc,[i for i,row in selected]));pointmap[m['marker_id']]=list(zip((xx+l).tolist(),(yy+t).tolist()))
 crop=im[box[1]:box[3],box[0]:box[2]];crop=cv2.resize(crop,None,fx=4,fy=4,interpolation=cv2.INTER_LANCZOS4);crop=cv2.copyMakeBorder(crop,32,32,32,32,cv2.BORDER_CONSTANT,value=(255,255,255));fn=o/'crops'/f"{m['marker_id']}_glyph_only4x.png";assert cv2.imwrite(str(fn),crop);jobs.append(dict(marker_id=m['marker_id'],page_id=m['page_id'],path=str(fn),source_box=box,width=crop.shape[1],height=crop.shape[0],pixels=crop.shape[0]*crop.shape[1]))
save(o/'glyph_jobs.json',jobs);raw=o/'glyph_raw.jsonl';assert not raw.exists()
with raw.open('w') as f,(o/'glyph_native.log').open('w') as err:subprocess.run([str(p/'code/recognize'),*(j['path'] for j in jobs)],stdout=f,stderr=err,check=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2'))
preds=[json.loads(s) for s in raw.read_text().splitlines()];assert len(preds)==17
rows=[]
for m,j,result in zip(labels,jobs,preds,strict=True):
 assert j['path']==result['path'];tokens=[dict(rank=c['rank'],text=numeric(v['text'])) for line in result['lines'] for c in line['candidates'] for v in c['matches']];rows.append(dict(marker_id=m['marker_id'],gold_identity=m['identity'],top1_exact=any(q['rank']==1 and q['text']==numeric(m['identity']) for q in tokens),top3_contains_exact=any(q['text']==numeric(m['identity']) for q in tokens),raw_top1=' | '.join(line['candidates'][0]['text'] for line in result['lines'] if line['candidates']),observation_count=len(result['lines']),elapsed_ms=result['elapsedMS'],pixels=j['pixels']))
save(o/'glyph_rows.json',rows)
spec=importlib.util.spec_from_file_location('old_method',old/'base/code/method.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.CONFIG['max_crops']=100000;ranked={}
for pg in pages:
 im=cv2.imread(pg['path'],0);ih,iw=im.shape;simple=[dict(text=l['text'],box=(np.array(l['box'])/[iw,ih,iw,ih]).tolist()) for l in pg['lines']];ranked[pg['page_id']]=module.proposals(im,simple)
stages=json.loads((o/'stage_rows.json').read_text())
for s in stages:
 props=ranked[s['page_id']];cov=[ink_coverage(pointmap[s['marker_id']],q['box']) for q in props];ranks=[i+1 for i,c in enumerate(cov) if c>=.98];s['glyph_only_first_complete_rank']=min(ranks) if ranks else None;s['glyph_only_selected_max_coverage']=max(cov[:32],default=0);s['glyph_only4x_exact']=next(r['top1_exact'] for r in rows if r['marker_id']==s['marker_id'])
save(o/'stage_rows_glyph_review.json',stages);summary=json.loads((o/'summary.json').read_text());summary['glyph_only4x_posthoc']=dict(calls=17,top1_exact=sum(r['top1_exact'] for r in rows),top3_contains_exact=sum(r['top3_contains_exact'] for r in rows),pixels=sum(j['pixels'] for j in jobs),elapsed_ms=sum(r['elapsed_ms'] for r in rows),gold_guided=True,posthoc=True);summary['glyph_only_complete_before_top32']=sum(s['glyph_only_first_complete_rank'] is not None for s in stages);summary['glyph_only_complete_excluded_by_top32']=sum(s['glyph_only_first_complete_rank'] is not None and s['glyph_only_first_complete_rank']>32 for s in stages);save(o/'summary.json',summary);print(json.dumps(summary,indent=2))
