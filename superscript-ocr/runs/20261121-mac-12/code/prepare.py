from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw
from lines import physical_lines,rectangle,tokens

P=Path(__file__).resolve().parents[1];O=P/'outputs'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def main():
    assert not (O/'gold_lines.json').exists()
    protocol=json.loads((P/'protocol.json').read_text());gt=json.loads(Path(protocol['gt_file']).read_text());byid={Path(a['page_info']['image_path']).stem:a for a in gt};pages=[];lines=[];shorts=[]
    downloads={r['case_id']:r for r in json.loads((P/'inputs/download_manifest.json').read_text())}
    for s in protocol['sample']:
        record=downloads[s['case_id']]
        if record['status']!='downloaded':continue
        fn=Path(record['path']);image=Image.open(fn).convert('RGB');assert image.size==(s['width'],s['height']);annotation=byid[s['page_id']]
        page=dict(s,path=str(fn),sha256=hashlib.sha256(fn.read_bytes()).hexdigest());pages.append(page)
        overlay=image.copy();d=ImageDraw.Draw(overlay)
        for block in annotation['layout_dets']:
            if block.get('ignore') or block['category_type']!='text_block':continue
            spans=block.get('line_with_spans',[])
            for g in physical_lines(spans):
                g.update(line_id=f'line-{len(lines)+1:04}',case_id=s['case_id'],block_id=block.get('anno_id'));lines.append(g);d.rectangle(g['box'],outline='red',width=1)
            for span in spans:
                if span['category_type']=='footnote_mark':shorts.append(dict(marker_id=f'mark-{len(shorts)+1:03}',case_id=s['case_id'],box=rectangle(span['poly']),raw=span.get('latex',span.get('text','')),block_id=block.get('anno_id')))
        overlay.thumbnail((1000,1500));overlay.save(O/(s['case_id']+'-annotation.png'))
        print(s['case_id'],'lines',sum(l['case_id']==s['case_id'] for l in lines),'markers',sum(l['case_id']==s['case_id'] for l in shorts),flush=True)
    save(P/'pages.json',pages);save(O/'gold_lines.json',lines);save(O/'gold_short_markers.json',shorts)

if __name__=='__main__':main()
