"""Rasterization and evaluation-only PDF/JATS alignment; no OCR predictions read."""
from pathlib import Path
import json, re, subprocess, xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from relations import citations, words
from gold import jats_relations

P=Path(__file__).resolve().parents[1]
O=P/'outputs'
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def overlap(a,b):
    a,b=set(a),set(b)
    return len(a&b)/max(1,min(len(a),len(b)))

def main():
    assert not (O/'gold.json').exists()
    pages=[]; gold=[]; exclusions=[]; references={}
    for source in json.loads((P/'sources.json').read_text()):
        sid=source['id'];d=P/'data'/sid;images=d/'pages';images.mkdir(exist_ok=True)
        subprocess.run(['pdftoppm','-r','150','-gray','-png',str(d/'source.pdf'),str(images/'page')],check=True,capture_output=True)
        xml=ET.parse(d/'evaluation_text.xml');ns={'x':'http://www.w3.org/1999/xhtml'}
        page_nodes=xml.findall('.//x:page',ns)
        for pi,p in enumerate(page_nodes):
            candidates=sorted(images.glob('page-*.png'));im=candidates[pi]
            with Image.open(im) as image: size=image.size
            pages.append(dict(id=f'{sid}-p{pi+1:02}',source=sid,page=pi+1,path=str(im),width=size[0],height=size[1],local=pi<4))
        source_xml=(d/'source.xml').read_text();relations,rejected=jats_relations(source_xml)
        exclusions.extend(dict(source=sid,**r) for r in rejected)
        root=ET.fromstring(source_xml)
        references[sid]={r.get('id'):dict(label=''.join(r.find('label').itertext()) if r.find('label') is not None else '',text=' '.join(r.itertext())) for r in root.findall('.//ref-list/ref')}
        pdf_groups=[]
        for pi,node in enumerate(page_nodes[:4]):
            tokens=[];chunks=[];offset=0
            for word in node.findall('.//x:word',ns):
                text=word.text or ''
                tokens.append(dict(start=offset,end=offset+len(text),box=[float(word.get(k)) for k in ['xMin','yMin','xMax','yMax']]))
                chunks.append(text);offset+=len(text)+1
            text=' '.join(chunks)
            for marker in citations(text):
                inside=[t for t in tokens if t['end']>marker['start'] and t['start']<marker['end']]
                box=[min(t['box'][0] for t in inside),min(t['box'][1] for t in inside),max(t['box'][2] for t in inside),max(t['box'][3] for t in inside)]
                pdf_groups.append(dict(**marker,page=pi+1,box_pt=box))
        assigned=set()
        for ri,relation in enumerate(relations):
            if relation.get('unbracketed'):
                exclusions.append(dict(source=sid,relation_index=ri,reason='unbracketed_outside_pilot_scope'));continue
            choices=[]
            for qi,q in enumerate(pdf_groups):
                if qi in assigned or q['numbers']!=relation['numbers']:continue
                left=overlap(words(relation['left'])[-6:],words(q['left'])[-6:]);right=overlap(words(relation['right'])[:6],words(q['right'])[:6])
                if left>=.5 and right>=.5:choices.append(((left+right)/2,qi,q))
            choices.sort(key=lambda a:-a[0])
            if not choices:
                exclusions.append(dict(source=sid,relation_index=ri,reason='not_unambiguously_aligned_on_first4pages'));continue
            if len(choices)>1 and choices[0][0]-choices[1][0]<.15:
                exclusions.append(dict(source=sid,relation_index=ri,reason='ambiguous_repeated_context'));continue
            score,qi,q=choices[0];assigned.add(qi);box=q['box_pt']
            if box[2]-box[0]>120 or box[3]-box[1]>25:
                exclusions.append(dict(source=sid,relation_index=ri,reason='multiline_or_oversized_marker'));continue
            page=next(p for p in pages if p['source']==sid and p['page']==q['page'])
            scale=page['width']/float(page_nodes[q['page']-1].get('width'))
            gold.append(dict(case_id=f'case-{len(gold)+1:03}',source=sid,page_id=page['id'],page=q['page'],numbers=relation['numbers'],rids=relation['rids'],left=relation['left'],right=relation['right'],box=[v*scale for v in box],alignment_score=score,paragraph=relation['paragraph'],visible=relation['visible'],review_required=True))
        print(sid,'pages',len(page_nodes),'JATS groups',len(relations),'aligned',sum(g['source']==sid for g in gold),flush=True)
    save(O/'pages.json',pages);save(O/'gold.json',gold);save(O/'gold_exclusions.json',exclusions);save(O/'gold_references.json',references)
    byid={p['id']:p for p in pages}
    for start in range(0,len(gold),10):
        sheet=Image.new('RGB',(1200,150*min(10,len(gold)-start)),'white');draw=ImageDraw.Draw(sheet)
        for row,g in enumerate(gold[start:start+10]):
            image=Image.open(byid[g['page_id']]['path']).convert('RGB');l,t,r,b=g['box'];cx=(l+r)/2;cy=(t+b)/2
            left=max(0,int(cx-285));top=max(0,int(cy-32));crop=image.crop((left,top,min(image.width,left+570),min(image.height,top+70)))
            cd=ImageDraw.Draw(crop);cd.rectangle((l-left,t-top,r-left,b-top),outline='red',width=1);crop=crop.resize((1140,140))
            sheet.paste(crop,(0,row*150+10));draw.text((5,row*150),f"{g['case_id']} {g['source']} p{g['page']} -> {g['numbers']} {g['rids']}",fill='blue')
        sheet.save(O/f'gold_review_{start//10:02}.png')

if __name__=='__main__':main()
