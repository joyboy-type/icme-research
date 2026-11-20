from pathlib import Path
import json,collections
from PIL import Image,ImageDraw
from lines import compare_line

P=Path(__file__).resolve().parents[1];O=P/'outputs'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def main():
    assert not (O/'line_screen.json').exists()
    pages=json.loads((P/'pages.json').read_text());gold=json.loads((O/'gold_lines.json').read_text());raw=[json.loads(s) for s in (O/'whole_raw.jsonl').read_text().splitlines()];evaluated=[];candidates=[]
    for page in pages:
        results=[r for r in raw if r['path']==page['path']];observations={}
        for result in results:
            if result['status']!='completed':continue
            obs=[]
            for line in result['lines']:
                if not line['candidates']:continue
                x,y,w,h=line['box'];obs.append(dict(text=line['candidates'][0]['text'],confidence=line['candidates'][0]['confidence'],box=[x*page['width'],y*page['height'],(x+w)*page['width'],(y+h)*page['height']]))
            observations[result['configuration']]=obs
        for line in [g for g in gold if g['case_id']==page['case_id']]:
            comparisons={name:compare_line(line,obs) for name,obs in observations.items()}
            missing=[name for name,r in comparisons.items() if r['screen_status']!='present'];absent=[name for name,r in comparisons.items() if r['screen_status']=='omission_candidate']
            row=dict(**line,comparisons=comparisons,nonpresent_configurations=missing,geometrically_absent_configurations=absent)
            evaluated.append(row)
            if len(missing)>=2:candidates.append(row)
    save(O/'line_screen.json',evaluated);save(O/'review_candidates.json',candidates)
    save(O/'screen_summary.json',dict(annotated_line_groups=len(gold),completed_requests=sum(r['status']=='completed' for r in raw),error_requests=sum(r['status']=='error' for r in raw),candidate_lines_shared_by_at_least2configurations=len(candidates),strict_geometric_absence_by_at_least2=sum(len(r['geometrically_absent_configurations'])>=2 for r in candidates),all3nonpresent=sum(len(r['nonpresent_configurations'])==3 for r in candidates),candidate_pages=dict(collections.Counter(c['case_id'] for c in candidates)),status='Candidates only; replacements, merged observations, and annotation errors require visual adjudication'))
    byid={p['case_id']:p for p in pages}
    for start in range(0,len(candidates),6):
        sheet=Image.new('RGB',(1500,260*min(6,len(candidates)-start)),'white');draw=ImageDraw.Draw(sheet)
        for row,c in enumerate(candidates[start:start+6]):
            image=Image.open(byid[c['case_id']]['path']).convert('RGB');l,t,r,b=c['box'];left=max(0,int(l-15));top=max(0,int(t-28));crop=image.crop((left,top,min(image.width,int(r+15)),min(image.height,int(b+28))));cd=ImageDraw.Draw(crop);cd.rectangle((l-left,t-top,r-left,b-top),outline='red',width=1);crop=crop.resize((crop.width*2,crop.height*2));crop.thumbnail((1460,145));sheet.paste(crop,(5,row*260+75))
            draw.text((5,row*260+2),c['line_id']+' '+c['case_id']+' GT: '+c['text'][:180],fill='blue')
            for j,(name,r) in enumerate(c['comparisons'].items()):draw.text((5,row*260+19+j*16),name+' '+r['screen_status']+' | '+' / '.join(r['nearest_texts'])[:165],fill='black')
        sheet.save(O/f'line_review_{start//6:02}.png')

if __name__=='__main__':main()
