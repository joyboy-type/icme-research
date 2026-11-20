"""Development-only gate revision; all context OCR is reused unchanged."""
from pathlib import Path
import json
from collections import Counter
from gate import localized_decision
p=Path(__file__).resolve().parents[1];o=p/'outputs'
jobs=json.loads((o/'context_jobs.json').read_text());results=[json.loads(s) for s in (o/'context_raw.jsonl').read_text().splitlines()]
idx={(j['source_run'],j['page_id'],j['method'],j['index']):(j,r) for j,r in zip(jobs,results,strict=True)}
patches=[]
for source in json.loads((p/'sources.json').read_text()):patches.extend(dict(q,source_run=source['id']) for q in json.loads(Path(source['patches']).read_text()))
decisions=[];accepted=[]
for q in patches:
    j,r=idx[q['source_run'],q['page_id'],q['method'],q['job_index']];ok,why,texts=localized_decision(q,j,r)
    decisions.append(dict(q,accepted=ok,decision_reason=why,reject_reason=None if ok else why,context_texts=texts))
    if ok:accepted.append(q)
(o/'decisions.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2));(o/'patches.json').write_text(json.dumps(accepted,ensure_ascii=False,indent=2))
for pg in json.loads((o/'pages.json').read_text()):
    for method in ('candidate','uniform'):
        text=[]
        for li,line in enumerate(pg['lines']):
            s=line['text']
            for a in sorted([a for a in accepted if a['page_id']==pg['page_id'] and a['method']==method and a['line_id']==li],key=lambda a:-a['offset']):
                k=a['offset'];s=s[:k]+'<sup>'+a['identity']+'</sup>'+s[k:]
            text.append(s)
        (o/'repaired'/method/(pg['page_id']+'.md')).write_text('\n'.join(text)+'\n')
s=json.loads((o/'gate_summary.json').read_text());s['patches_after']={m:sum(q['method']==m for q in accepted) for m in ['candidate','uniform']};s['rejections']=dict(Counter(q['decision_reason'] for q in decisions if not q['accepted']));s['development_revision']='merge body and superscript observations on same source row; initial strict per-observation rule preserved in outputs/initial/; no new OCR';(o/'gate_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
print([(q['source_run'][-2:],q['text'],q['accepted'],q['decision_reason'],q['context_texts']) for q in decisions])
