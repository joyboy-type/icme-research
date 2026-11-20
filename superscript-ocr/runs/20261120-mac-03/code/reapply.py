"""Development attachment revision: reuses saved OCR without any new OCR calls."""
from pathlib import Path
import json
from method import attach
P=Path(__file__).resolve().parents[1];O=P/'outputs';V=O/'attachment_v2';V.mkdir(exist_ok=True)
pages=json.loads((O/'pages.json').read_text());pred=json.loads((O/'predictions.json').read_text());patches=[]
for page in pages:
    for method in ('candidate','uniform'):
        current=attach(page['lines'],[p for p in pred if p['page_id']==page['page_id'] and p['method']==method],page['numbers']);patches.extend(current)
        text=[]
        for i,line in enumerate(page['lines']):
            s=line['text']
            for patch in sorted([p for p in current if p['line_id']==i],key=lambda p:-p['offset']):
                k=patch['offset'];s=s[:k]+f"<sup>{patch['identity']}</sup>"+s[k:]
            text.append(s)
        folder=V/'repaired'/method;folder.mkdir(parents=True,exist_ok=True)
        (folder/(page['page_id']+'.md')).write_text('\n'.join(text)+'\n')
(V/'patches.json').write_text(json.dumps(patches,ensure_ascii=False,indent=2));print(len(patches),'patches')
