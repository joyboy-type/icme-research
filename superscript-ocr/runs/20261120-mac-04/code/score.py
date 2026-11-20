from pathlib import Path
import json, sys, re, unicodedata, difflib, hashlib
from PIL import Image, ImageDraw, ImageFont
from method import numeric

ROOT=Path(__file__).resolve().parents[5]

def at_marker(p,m,require_number=True):
    poly=m['span']['poly'];l,r=min(poly[::2]),max(poly[::2]);t,b=min(poly[1::2]),max(poly[1::2])
    mx,my=max(5,(r-l)*.2),max(5,(b-t)*.2)
    x,y=p['xy']
    return (not require_number or numeric(p['text'])==numeric(m['identity'])) and l-mx<=x<=r+mx and t-my<=y<=b+my

def norm(s):return ''.join(c.lower() for c in unicodedata.normalize('NFKC',s) if c.isalnum())

def anchor_ok(p,m):
    left,right=norm(p['left']),norm(p['right']);before,after=norm(m.get('anchor_before','')),norm(m.get('anchor_after',''))
    n=min(16,len(left),len(before))
    if n<6:return False
    if difflib.SequenceMatcher(None,left[-n:],before[-n:]).ratio()<.85:return False
    n=min(12,len(right),len(after))
    return n<6 or difflib.SequenceMatcher(None,right[:n],after[:n]).ratio()>=.85

def enrich(markers):
    gt=json.loads((ROOT/'work/textscreen_gate_20261120/gt.json').read_text());byid={Path(r['page_info']['image_path']).stem:r for r in gt}
    out=[]
    for raw in markers:
        m=dict(raw);s=m['span'].get('latex','');clean=s.replace('$','').replace('^','').replace('{','').replace('}','').strip()
        if not re.fullmatch(r'[\d\s,，\[\]–—\-]+',clean):continue
        m.setdefault('identity',re.sub(r'\s+','',clean).replace('，',',').replace('–','-').replace('—','-'))
        key=(m['page_id'],tuple(round(x,2) for x in m['span']['poly']))
        m.setdefault('marker_id',hashlib.sha256(repr(key).encode()).hexdigest()[:12])
        block=next((b for b in byid[m['page_id']]['layout_dets'] if b.get('anno_id')==m['block_id']),None)
        if block:
            spans=block.get('line_with_spans',[]);i=m['span_i']
            m['anchor_before']=' '.join(x.get('text',x.get('latex','')) for x in spans[:i])[-250:]
            m['anchor_after']=' '.join(x.get('text',x.get('latex','')) for x in spans[i+1:])[:250]
        out.append(m)
    return out

def main():
    p=Path(sys.argv[1]).resolve();o=p/'outputs';labels=enrich(json.loads(Path(sys.argv[2]).read_text()))
    pages=json.loads((o/'pages.json').read_text());pageids={p['page_id'] for p in pages}
    labels=[m for m in labels if m['page_id'] in pageids];assert labels
    allmarks=enrich(json.loads((ROOT/'experiments/superscript-ocr/data/snapshot-01/annotated_markers.json').read_text()))
    allmarks=[m for m in allmarks if m['page_id'] in pageids]
    preds=json.loads((o/'predictions.json').read_text())
    patchdir=o/sys.argv[3] if len(sys.argv)>3 else o
    patches=json.loads((patchdir/'patches.json').read_text())
    rows=[]
    for m in labels:
        page=next(p for p in pages if p['page_id']==m['page_id']);row=dict(marker_id=m['marker_id'],page_id=m['page_id'],identity=m['identity'])
        row['full_hit']=any(at_marker(t,m) for t in page['numbers'])
        for method in ('candidate','uniform'):
            pp=[t for t in preds if t['method']==method and t['page_id']==m['page_id']]
            patches_m=[t for t in patches if t['method']==method and t['page_id']==m['page_id']]
            poly=m['span']['poly'];cx=(min(poly[::2])+max(poly[::2]))/2;cy=(min(poly[1::2])+max(poly[1::2]))/2
            jobs=json.loads((o/'jobs.json').read_text())
            row[method+'_covered']=any(j['box'][0]<=cx<=j['box'][2] and j['box'][1]<=cy<=j['box'][3] for j in jobs if j['page_id']==m['page_id'] and j['method']==method)
            row[method+'_hit']=any(at_marker(t,m) for t in pp)
            row[method+'_inserted_at_position']=any(at_marker(t,m) for t in patches_m)
            row[method+'_inserted_at_anchor']=any(at_marker(t,m) and anchor_ok(t,m) for t in patches_m)
            row[method+'_union_hit']=row['full_hit'] or row[method+'_inserted_at_anchor']
        rows.append(row)
    audit=[]
    for i,patch in enumerate(patches):
        nearby=[m for m in allmarks if m['page_id']==patch['page_id'] and at_marker(patch,m,False)]
        exact=[m for m in nearby if at_marker(patch,m)]
        ok=[m for m in exact if anchor_ok(patch,m)]
        audit.append(dict(index=i,**patch,annotation_status='exact_number_and_anchor' if ok else 'anchor_unverified' if exact else 'wrong_number' if nearby else 'unmatched_needs_pixel_review',nearby_gold=[dict(marker_id=m['marker_id'],identity=m['identity'],before=m.get('anchor_before'),after=m.get('anchor_after')) for m in nearby]))
    summary=dict(markers=len(labels),pages=len(pages),full_hit=sum(r['full_hit'] for r in rows),methods={},scope='Known marker number+image-position scoring; automatic text-anchor check only for insertions. Full baseline anchor not independently scored. All inserted groups require pixel audit, especially unmatched annotations.')
    for method in ('candidate','uniform'):
        summary['methods'][method]={k:sum(r[method+'_'+k] for r in rows) for k in ('covered','hit','inserted_at_position','inserted_at_anchor','union_hit')}
        summary['methods'][method]['patches']=sum(a['method']==method for a in audit)
    for name,value in [('per_marker',rows),('insertion_audit',audit),('summary',summary),('evaluation_labels',labels)]:
        (patchdir/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    # Every inserted group is displayed with its source location and actual predicted text.
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
    for start in range(0,len(audit),8):
        sheet=Image.new('RGB',(1300,230*min(8,len(audit)-start)),'#eeeeee');draw=ImageDraw.Draw(sheet)
        for row,a in enumerate(audit[start:start+8]):
            page=next(q for q in pages if q['page_id']==a['page_id']);im=Image.open(page['path']).convert('RGB');x,y=a['xy'];h=page['lines'][a['line_id']]['box'][3]
            l=max(0,int(x-230));t=max(0,int(y-h*1.0));r=min(im.width,int(x+180));b=min(im.height,int(y+h*2.2))
            crop=im.crop((l,t,r,b));crop=crop.resize((int(crop.width*2),int(crop.height*2)))
            if crop.height>155:crop.thumbnail((1000,155))
            sheet.paste(crop,(8,row*230+65))
            draw.text((8,row*230+4),f"#{a['index']} {a['method']} -> {a['text']} | {a['annotation_status']} | {a['page_id'][-65:]}",font=font,fill='black')
            draw.text((8,row*230+27),(a['left'][-70:]+' <'+a['text']+'> '+a['right'][:60])[:135],font=font,fill='black')
        sheet.save(patchdir/f'insertion_review_{start//8:02}.png')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
