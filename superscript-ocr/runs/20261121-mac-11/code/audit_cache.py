"""Read-only fixed-batch screening of cached OCR against source annotations."""
from pathlib import Path
import json, hashlib, collections
from PIL import Image, ImageDraw
from omissions import tokens, footnote_tokens, anchored_gap, common_deletions, normal_looking, align

P=Path(__file__).resolve().parents[1];O=P/'outputs'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def box(poly):return [min(poly[::2]),min(poly[1::2]),max(poly[::2]),max(poly[1::2])]

def main():
    assert not (O/'screen_candidates.json').exists()
    protocol=json.loads((P/'protocol.json').read_text());gt=json.loads(Path(protocol['gt_file']).read_text());index={Path(g['page_info']['image_path']).stem:g for g in gt}
    pages=[];candidates=[];shorts=[]
    for source in json.loads((P/'sources.json').read_text()):
        for kind,path in source['paths'].items():assert sha(Path(path))==source['sha256'][kind]
        pid=source['page_id'];annotation=index[pid];image=Image.open(source['paths']['image'])
        assert image.size==(annotation['page_info']['width'],annotation['page_info']['height'])
        texts={m:Path(source['paths'][m]).read_text() for m in ['olmo2','deepseek']}
        q=json.loads(Path(source['paths']['qwen']).read_text());texts['qwen']=''.join(t['token'] for t in q['tokens'])
        predictions={m:tokens(text) for m,text in texts.items()}
        sequence=[];spans=[];blocks=[]
        eligible=[b for b in annotation['layout_dets'] if not b.get('ignore') and b['category_type']=='text_block' and b.get('order') is not None]
        for block in sorted(eligible,key=lambda b:b['order']):
            text=block.get('text','');nalpha=sum(c.isascii() and c.isalpha() for c in text)
            if nalpha<20:continue
            begin=len(sequence);parts=block.get('line_with_spans') or [dict(text=text,poly=block['poly'],category_type='text_span')]
            for si,span in enumerate(parts):
                raw=span.get('text',span.get('latex',''));ts=footnote_tokens(raw) if span['category_type']=='footnote_mark' else tokens(raw);start=len(sequence);sequence.extend(ts)
                spans.append(dict(start=start,end=len(sequence),tokens=ts,category=span['category_type'],box=box(span.get('poly',block['poly'])),block_id=block.get('anno_id'),span_index=si))
            blocks.append(dict(start=begin,end=len(sequence),block_id=block.get('anno_id'),box=box(block['poly'])))
        runs=common_deletions(sequence,list(predictions.values()));shared={i for a,b in runs for i in range(a,b)}
        page=dict(case_id=source['case_id'],page_id=pid,image=source['paths']['image'],source_attribute=annotation['page_info']['page_attribute'],eligible_blocks=len(blocks),eligible_tokens=len(sequence),numeric_footnote_spans=sum(s['category']=='footnote_mark' and s['tokens'] and all(t.isdigit() for t in s['tokens']) for s in spans),normal_looking={m:normal_looking(v) for m,v in predictions.items()},prediction_tokens={m:len(v) for m,v in predictions.items()},qwen_mean_nll=-sum(t['logprob'] for t in q['tokens'])/len(q['tokens']),common_deleted_tokens=len(shared))
        pages.append(page)
        for a,b in runs:
            overlapping=[s for s in spans if s['end']>a and s['start']<b]
            if not overlapping:continue
            boxes=[s['box'] for s in overlapping];bbox=[min(x[0] for x in boxes),min(x[1] for x in boxes),max(x[2] for x in boxes),max(x[3] for x in boxes)]
            contexts={}
            for m,prediction in predictions.items():
                ops=align(sequence,prediction);op=next((op for op in ops if op[1]<=a<op[2]),None);pos=op[3] if op else 0
                contexts[m]=' '.join(prediction[max(0,pos-25):pos+25])
            kind='long_lexical_run' if b-a>=8 else 'short_numeric' if all(t.isdigit() for t in sequence[a:b]) else 'short_lexical_run'
            candidates.append(dict(candidate_id=f'cand-{len(candidates)+1:03}',page_id=pid,page_case=source['case_id'],start=a,end=b,length=b-a,type=kind,text=' '.join(sequence[a:b]),before=' '.join(sequence[max(0,a-25):a]),after=' '.join(sequence[b:b+25]),box=bbox,model_contexts=contexts,all_normal_looking=all(page['normal_looking'].values()),manual_review_required=True))
        for s in spans:
            if s['category']!='footnote_mark' or not s['tokens'] or not all(t.isdigit() for t in s['tokens']):continue
            gaps={m:anchored_gap(sequence[max(0,s['start']-8):s['start']],sequence[s['end']:s['end']+8],prediction) for m,prediction in predictions.items()}
            statuses={m:'unresolved_anchor' if gap is None else 'omitted' if not gap else 'correct' if [t for t in gap if t.isdigit()]==s['tokens'] else 'misread_or_other_content' for m,gap in gaps.items()}
            shorts.append(dict(page_id=pid,block_id=s['block_id'],span_index=s['span_index'],tokens=s['tokens'],box=s['box'],shared_deleted=all(i in shared for i in range(s['start'],s['end'])),gaps=gaps,statuses=statuses,all_anchor_evaluable=all(g is not None for g in gaps.values()),all_omitted=all(v=='omitted' for v in statuses.values()),all_not_correct=all(v in ['omitted','misread_or_other_content'] for v in statuses.values()),before=' '.join(sequence[max(0,s['start']-18):s['start']]),after=' '.join(sequence[s['end']:s['end']+18])))
        print(source['case_id'],page['eligible_blocks'],page['eligible_tokens'],'shared deleted',len(shared),'long runs',sum(c['page_id']==pid and c['type']=='long_lexical_run' for c in candidates),flush=True)
    save(O/'pages.json',pages);save(O/'screen_candidates.json',candidates);save(O/'short_symbol_screen.json',shorts)
    save(O/'screen_summary.json',dict(pages=len(pages),eligible_blocks=sum(p['eligible_blocks'] for p in pages),eligible_tokens=sum(p['eligible_tokens'] for p in pages),numeric_footnote_spans=len(shorts),all_normal_looking_pages=sum(all(p['normal_looking'].values()) for p in pages),candidate_runs=len(candidates),candidate_types=dict(collections.Counter(c['type'] for c in candidates)),short_spans_shared_deleted=sum(s['shared_deleted'] for s in shorts),short_spans_all_anchors_evaluable=sum(s['all_anchor_evaluable'] for s in shorts),short_spans_all_omitted=sum(s['all_omitted'] for s in shorts),short_spans_all_not_correct=sum(s['all_not_correct'] for s in shorts),truth_status='unreviewed alignment candidates, not confirmed omissions',finish_reasons_available=False,new_ocr_calls=0))
    byid={s['page_id']:s for s in json.loads((P/'sources.json').read_text())}
    for start in range(0,len(candidates),6):
        sheet=Image.new('RGB',(1400,260*min(6,len(candidates)-start)),'white');d=ImageDraw.Draw(sheet)
        for row,c in enumerate(candidates[start:start+6]):
            image=Image.open(byid[c['page_id']]['paths']['image']).convert('RGB');l,t,r,b=c['box'];crop=image.crop((max(0,int(l-15)),max(0,int(t-15)),min(image.width,int(r+15)),min(image.height,int(b+15))));crop.thumbnail((1360,190));sheet.paste(crop,(5,row*260+55));d.text((5,row*260+3),c['candidate_id']+' '+c['page_case']+' '+c['type']+' '+c['text'][:140],fill='blue');d.text((5,row*260+23),'BEFORE '+c['before'][-130:],fill='black');d.text((5,row*260+38),'AFTER '+c['after'][:130],fill='black')
        sheet.save(O/f'screen_review_{start//6:02}.png')

if __name__=='__main__':main()
