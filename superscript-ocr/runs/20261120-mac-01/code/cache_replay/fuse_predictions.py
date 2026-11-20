# Inference reads only public OCR predictions. No annotations, source PDFs, source text layer or gold boxes.
from pathlib import Path
import re,json,hashlib,unicodedata
from bs4 import BeautifulSoup
P=Path(__file__).parent;(P/'fusion').mkdir(exist_ok=True)
def canonical(s):
 ignored=set()
 for m in re.finditer(r'\\[a-zA-Z]+',s):ignored.update(range(m.start(),m.end()))
 c=[];mapping=[]
 for i,char in enumerate(s):
  if i in ignored:continue
  for z in unicodedata.normalize('NFKC',char).lower():
   if z.isalnum():c.append(z);mapping.append(i)
 return ''.join(c),mapping
def findends(s,a,side):
 aa=canonical(a)[0]
 for n in range(min(64,len(aa)),15,-1):
  needle=aa[-n:] if side=='left' else aa[:n];hits=[m.start() for m in re.finditer(re.escape(needle),s)]
  if len(hits)==1:return (hits[0]+n if side=='left' else hits[0]),n
 return None,0
logs=[]
# Input list comes from available secondary predictions; does not inspect marker annotations.
for f in sorted((P/'olmo2').glob('*.md')):
 id=f.stem;cached=P/'cache'/(hashlib.sha256(id.encode()).hexdigest()[:16]+'.json')
 if not cached.exists():continue
 raw=''.join(t['token'] for t in json.loads(cached.read_text())['tokens']);soup=BeautifulSoup(raw,'html.parser')
 for e in soup.select('script,style,img,div.image'):e.decompose()
 base=soup.get_text(' ',strip=True).replace('```html','').replace('```','')
 other=BeautifulSoup(f.read_text(),'html.parser').get_text(' ',strip=True)
 target,positions=canonical(base);patches=[]
 # Fixed punctuation-adjacent 1–3 digit citation syntax. Explicitly exclude decimals and digit-prefix list commas.
 pattern=r'(?<!\d)([.!?;,])(?P<id>\d{1,3}(?:[,–-]\d{1,3})*)(?=\s|$)'
 for m in re.finditer(pattern,other):
  l,ls=findends(target,other[:m.start('id')],'left');r,rs=findends(target,other[m.end('id'):],'right')
  if l is None or r is None or r!=l:continue
  # A zero-alphanumeric gap is the only repair target; existing numbers or words cannot be replaced.
  start=positions[l-1]+1 if l else 0;end=positions[r] if r<len(positions) else len(base);gap=base[start:end]
  if not re.fullmatch(r'[\s.!?;,"“”‘’\'()\[\]{}$\\^`*_\-]*',gap):continue
  value=m.group('id').replace('–','-')
  # Insert before following lexical token, after existing punctuation, preserving body bytes.
  insertion=end
  if any(pos==insertion for pos,_,_ in patches):continue
  patches.append((insertion,value,{'page_id':id,'secondary_value':value,'base_gap':gap,'secondary_context':other[max(0,m.start()-75):m.end()+75],'base_context':base[max(0,start-75):end+75],'left_anchor_chars':ls,'right_anchor_chars':rs}))
 fused=base
 for pos,value,info in sorted(patches,reverse=True):fused=fused[:pos]+'$^{'+value+'}$ '+fused[pos:];logs.append(info)
 (P/'fusion'/(id+'.txt')).write_text(fused)
(P/'fusion_patch_log.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2));print('patches',len(logs));print([(x['page_id'],x['secondary_value'],x['base_context']) for x in logs])
