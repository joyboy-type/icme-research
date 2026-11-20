from pathlib import Path
import re,json,hashlib,unicodedata,difflib
from bs4 import BeautifulSoup
P=Path(__file__).parent;sample=json.loads((P/'sample.json').read_text())
def canonmap(s):
 ignored=set()
 for m in re.finditer(r'\\[a-zA-Z]+',s):ignored.update(range(m.start(),m.end()))
 chars=[];pos=[]
 for i,c in enumerate(s):
  if i in ignored:continue
  for cc in unicodedata.normalize('NFKC',c).lower():
   if cc.isalnum():chars.append(cc);pos.append(i)
 return ''.join(chars),pos
def canon(s):return canonmap(s)[0]
def hits(s,q):return [m.start() for m in re.finditer(re.escape(q),s)] if q else []
def anchor(s,q,side):
 q=canon(q)
 for n in range(min(70,len(q)),11,-1):
  part=q[-n:] if side=='left' else q[:n];h=hits(s,part)
  if h:return [(i+n if side=='left' else i,n) for i in h]
 return []
def getid(raw):
 raw=unicodedata.normalize('NFKC',raw).replace('–','-').replace('—','-').replace('−','-').replace('，',',')
 raw=re.sub(r'\\[a-zA-Z]+','',raw);raw=raw.replace(' ','').replace('\n','')
 # Preserve commas and ranges; ignore wrapping brackets and superscript representation.
 vals=re.findall(r'\d+(?:[,\-]\d+)*',raw)
 return vals
texts={}
for id in sorted({m['page_id'] for m in sample}):
 a=json.loads((P/'cache'/(hashlib.sha256(id.encode()).hexdigest()[:16]+'.json')).read_text());raw=''.join(x['token'] for x in a['tokens']);s=BeautifulSoup(raw,'html.parser')
 for el in s.select('script,style,img,div.image'):el.decompose()
 texts[('qwen122',id)]=s.get_text(' ',strip=True).replace('```html','').replace('```','')
 for model in ['olmo2','deepseek']:
  f=P/model/(id+'.md')
  if f.exists():
   t=f.read_text();t=re.sub(r'!\[[^\]]*\]\([^)]*\)','',t);texts[(model,id)]=BeautifulSoup(t,'html.parser').get_text(' ',strip=True)
for f in (P/'fusion').glob('*.txt'):texts[('fusion',f.stem)]=f.read_text()
out=[]
for i,m in enumerate(sample):
 e={'index':i,'marker_id':m['marker_id'],'page_id':m['page_id'],'gold':m['identity'],'gold_numeric':getid(m['identity'])[0],'before':m['anchor_before'],'after':m['anchor_after'],'models':{}}
 for model in ['qwen122','olmo2','deepseek','fusion']:
  t=texts.get((model,m['page_id']))
  if t is None:e['models'][model]={'status':'cache_missing'};continue
  s,positions=canonmap(t);left=anchor(s,m['anchor_before'],'left');right=anchor(s,m['anchor_after'],'right');c=[]
  for l,ln in left:
   for r,rn in right:
    if 0<=r-l<=35:c.append((ln+rn,l,r,'both'))
   if not m['anchor_after'].strip():
    z=re.match(r'\d*',s[l:]);c.append((ln,l,l+len(z.group()),'left_only'))
  if not c:
   e['models'][model]={'status':'alignment_unresolved','left':left,'right':right,'model_text':t};continue
  c.sort(reverse=True);score,l,r,mode=c[0]
  # With adjacent anchors, original gap includes marker punctuation/markup. For left-only, use explicit nearby digits.
  start=positions[l-1]+1 if l else 0;end=positions[r] if r<len(positions) else len(t)
  raw=t[start:end] if mode=='both' else t[start:(positions[r-1]+1 if r>l else start+18)]
  vals=getid(raw)
  if mode=='left_only' and r==l:vals=[]
  identity=vals[0] if len(vals)==1 else '|'.join(vals)
  if identity==e['gold_numeric']:status='identity_at_anchor'
  elif not vals:status='omitted_at_anchor'
  else:status='wrong_identity_at_anchor'
  e['models'][model]={'status':status,'pred':identity,'raw_gap':raw,'alignment':mode,'matching_anchor_chars':score,'n_best_alignment_ties':sum(cc[0]==score for cc in c),'left_pos':l,'right_pos':r,'context':t[max(0,start-90):min(len(t),end+90)]}
 out.append(e)
(P/'automatic_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
from collections import Counter
print({model:dict(Counter(x['models'][model]['status'] for x in out)) for model in ['qwen122','olmo2','deepseek','fusion']})
for x in out:print(x['index'],x['gold_numeric'],{k:{a:b for a,b in v.items() if a in ['status','pred','raw_gap','alignment','n_best_alignment_ties']} for k,v in x['models'].items()})
