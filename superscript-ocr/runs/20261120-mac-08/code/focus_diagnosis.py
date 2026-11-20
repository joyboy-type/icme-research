"""Post-run evaluation of the focus17 proposal pool; no additional OCR or tuning."""
from pathlib import Path
import json,cv2,numpy as np
import baseline
from candidates import isolate,rerank
P=Path(__file__).resolve().parents[1];O=P/'outputs'
pages=json.loads((O/'pages.json').read_text());labels=[m for m in json.loads((O/'evaluation_labels.json').read_text()) if m['split']=='focus17'];glyph={j['marker_id']:j['source_box'] for j in json.loads((P.parent/'20261120-mac-07/outputs/glyph_jobs.json').read_text())};scores={r['marker_id']:r for r in json.loads((O/'per_marker.json').read_text())};out=[]
for pg in [p for p in pages if p['source_run']=='20261120-mac-06']:
 gray=cv2.cvtColor(cv2.imread(pg['path']),cv2.COLOR_BGR2GRAY);ih,iw=gray.shape;lines=[dict(text=l['text'],box=(np.array(l['box'])/[iw,ih,iw,ih]).tolist()) for l in pg['lines']];baseline.CONFIG['max_crops']=100000;pool=baseline.proposals(gray,lines);mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12);_,cc,stats,_=cv2.connectedComponentsWithStats(mask);new=rerank(pool,cc,stats,100000)
 for m in [m for m in labels if m['page_id']==pg['page_id']]:
  l,t,r,b=glyph[m['marker_id']];yy,xx=np.where(gray[t:b,l:r]<180);xx=xx+l;yy=yy+t
  def covers(box):
   x0,y0,x1,y1=box;return float(((xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)).mean())
  oldranks=[i+1 for i,p in enumerate(pool) if covers(p['box'])>=.98];newranks=[i+1 for i,p in enumerate(new) if covers(p['box'])>=.98];row=scores[m['marker_id']]
  out.append(dict(marker_id=m['marker_id'],page_id=m['page_id'],original_first_complete_rank=min(oldranks) if oldranks else None,new_first_complete_rank=min(newranks) if newranks else None,new_selected_complete=bool(newranks and min(newranks)<=32),isolate_number_hit=row['isolate_hit'],inserted_at_anchor=row['isolate_inserted_at_anchor'],note='Digit-box ink coverage is an evaluation proxy, not necessarily retained rendered ink. All candidate inputs remain label-free.'))
(O/'focus_diagnosis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('focus17: new-selected complete',sum(r['new_selected_complete'] for r in out),'pool lost',sum(r['original_first_complete_rank'] is not None and r['new_first_complete_rank'] is None for r in out),'still ranked out',sum(r['new_first_complete_rank'] is not None and r['new_first_complete_rank']>32 for r in out))
