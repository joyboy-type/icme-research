from pathlib import Path
import json,math,csv
from collections import Counter
from PIL import Image,ImageDraw,ImageFont
from evaluation import at_marker,anchor_ok,enrich
P=Path(__file__).resolve().parents[1];O=P/'outputs';R=P.parent
METHODS=['baseline','rerank','isolate','uniform']
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))

def main():
    assert len((O/'raw.jsonl').read_text().splitlines())==3072,'inference must finish first'
    labels=[]
    for rid,rel,split in [('20261120-mac-03','outputs/attachment_v2/evaluation_labels.json','development40'),('20261120-mac-04','outputs/evaluation_labels.json','previous33'),('20261120-mac-06','base/outputs/evaluation_labels.json','focus17')]:
        labels.extend(dict(m,split=split) for m in json.loads((R/rid/rel).read_text()))
    assert len(labels)==90
    pred=json.loads((O/'predictions.json').read_text());patches=json.loads((O/'patches.json').read_text());jobs=json.loads((O/'jobs.json').read_text());pages=json.loads((O/'pages.json').read_text());byid={p['page_id']:p for p in pages}
    rows=[]
    for m in labels:
        row=dict(marker_id=m['marker_id'],page_id=m['page_id'],identity=m['identity'],split=m['split']);poly=m['span']['poly'];x=(min(poly[::2])+max(poly[::2]))/2;y=(min(poly[1::2])+max(poly[1::2]))/2
        for method in METHODS:
            pp=[q for q in pred if q['page_id']==m['page_id'] and q['method']==method];aa=[q for q in patches if q['page_id']==m['page_id'] and q['method']==method]
            row[method+'_center_covered']=any(j['source_box'] is not None and j['source_box'][0]<=x<j['source_box'][2] and j['source_box'][1]<=y<j['source_box'][3] for j in jobs if j['page_id']==m['page_id'] and j['method']==method)
            row[method+'_hit']=any(at_marker(q,m) for q in pp)
            row[method+'_inserted_at_position']=any(at_marker(q,m) for q in aa)
            row[method+'_inserted_at_anchor']=any(at_marker(q,m) and anchor_ok(q,m) for q in aa)
        rows.append(row)
    save(O/'evaluation_labels.json',labels);save(O/'per_marker.json',rows)
    with (O/'per_marker.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(scope='All90reference positions on24already-seen pages are development data. Recognition number+position is primary crop metric. Insertion metrics are pre-context-gate and require source review. Matched canvas baseline is rerun, not interchangeable with old archived scores.',splits={})
    for split in ['development40','previous33','focus17','all90']:
        rr=[r for r in rows if split=='all90' or r['split']==split];summary['splits'][split]={'markers':len(rr),'methods':{m:{k:sum(r[m+'_'+k] for r in rr) for k in ['center_covered','hit','inserted_at_position','inserted_at_anchor']} for m in METHODS}}
    allmarks=enrich(json.loads((P.parents[1]/'data/snapshot-01/annotated_markers.json').read_text()))
    # Known nonreferences are an evaluation inventory only, never consulted in inference.
    nonrefs=[b['annotation'] for b in json.loads((R/'20261120-mac-05/inputs/negative_bank.json').read_text()) if 'annotation' in b]
    audit=[]
    for i,a in enumerate(patches):
        nearby=[m for m in allmarks if m['page_id']==a['page_id'] and at_marker(a,m,False)]
        exact=[m for m in nearby if at_marker(a,m)];valid=[m for m in exact if anchor_ok(a,m)]
        negative=[m for m in nonrefs if m['page_id']==a['page_id'] and at_marker(a,m,False)]
        status='known_nonreference' if negative else 'number_and_anchor_match' if valid else 'anchor_needs_review' if exact else 'wrong_number_needs_review' if nearby else 'unmatched_needs_review'
        audit.append(dict(index=i,**a,annotation_status=status,nearby_gold=[dict(identity=m['identity'],marker_id=m['marker_id'],before=m.get('anchor_before'),after=m.get('anchor_after')) for m in nearby]))
    save(O/'insertion_audit.json',audit)
    summary['all_proposed_insertions']={m:sum(a['method']==m for a in audit) for m in METHODS};summary['insertion_review_pending']=True;save(O/'summary.json',summary)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
    for start in range(0,len(audit),7):
        sheet=Image.new('RGB',(1250,230*min(7,len(audit)-start)),'#eee');d=ImageDraw.Draw(sheet)
        for row,a in enumerate(audit[start:start+7]):
            pg=byid[a['page_id']];im=Image.open(pg['path']).convert('RGB');x,y=a['xy'];h=pg['lines'][a['line_id']]['box'][3];l=max(0,int(x-190));t=max(0,int(y-h*.8));r=min(im.width,int(x+170));b=min(im.height,int(y+h*1.8));crop=im.crop((l,t,r,b));crop=crop.resize((crop.width*2,crop.height*2));crop.thumbnail((1000,150));sheet.paste(crop,(5,row*230+65));d.text((5,row*230+4),f"#{a['index']} {a['method']} -> {a['text']} | {a['annotation_status']} | {a['page_id'][-55:]}",font=font,fill='black');d.text((5,row*230+28),(a['left'][-65:]+' <'+a['text']+'> '+a['right'][:55])[:135],font=font,fill='black')
        sheet.save(O/f'insertion_review_{start//7:02}.png')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
