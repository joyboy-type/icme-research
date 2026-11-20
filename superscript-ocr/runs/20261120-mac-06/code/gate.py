"""Conservative context confirmation. No annotation or source-document text input."""
import re, unicodedata, difflib

CONFIG=dict(context_scale=4,context_padding=32,context_width_min=180,
            context_width_max=420,context_extra_width=180,context_height_min=36,
            context_extra_height=12,anchor_chars=12,anchor_similarity=.75,
            right_similarity=.70,minimum_anchor_chars=4)
NUM=re.compile(r'[0-9]+(?:[,，\-–−][0-9]+)*')

def canon(s):
    return unicodedata.normalize('NFKC',str(s)).translate(str.maketrans('–−—，','---,'))

def norm(s):return ''.join(c.lower() for c in canon(s) if c.isalnum())

def context_box(box,width,height):
    l,t,r,b=box;w=min(width,CONFIG['context_width_max'],max(CONFIG['context_width_min'],r-l+CONFIG['context_extra_width']))
    h=min(height,max(CONFIG['context_height_min'],b-t+CONFIG['context_extra_height']))
    x=min(max(0,round((l+r-w)/2)),width-w)
    y=min(max(0,round((t+b-h)/2+(b-t)*.20)),height-h)
    return [int(x),int(y),int(w),int(h)]

def decide(p,text):
    text=canon(text);identity=canon(p['text']).strip('[]')
    matches=[m for m in NUM.finditer(text) if m.group()==identity]
    if not matches:return False,'number_disagreement'
    reasons=[]
    for m in matches:
        before=text[:m.start()];after=text[m.end():]
        # Units and isotope notation are not citations, even when both OCR views agree.
        left=before.rstrip();right=after.lstrip()
        unit=bool(re.search(r'(?:/\s*[a-zA-Z]|\b(?:mm|cm|km|nm|m|s|kg|mol|GeV|MeV|eV))\s*$',left))
        isotope=bool(re.match(r'(?:C|H|N|O|Si|He|P)\b',right) and identity in {'1','2','3','4','12','13','14','15','16','17','18','29','31'})
        if unit or isotope:
            reasons.append('unit_or_isotope');continue
        a,b=norm(p.get('left','')),norm(before);n=min(CONFIG['anchor_chars'],len(a),len(b))
        if n<CONFIG['minimum_anchor_chars'] or difflib.SequenceMatcher(None,a[-n:],b[-n:]).ratio()<CONFIG['anchor_similarity']:
            reasons.append('left_anchor_disagreement');continue
        a,b=norm(p.get('right','')),norm(after);n=min(8,len(a),len(b))
        if n>=4 and difflib.SequenceMatcher(None,a[:n],b[:n]).ratio()<CONFIG['right_similarity']:
            reasons.append('right_anchor_disagreement');continue
        bracket=bool(re.search(r'[\[(]\s*$',before) and re.match(r'^\s*[\])]',after))
        sentence=bool(re.search(r'[.!?。！？;；:：)）]\s*$',before))
        if not (bracket or sentence):
            reasons.append('no_reference_format_evidence');continue
        return True,'context_number_anchor_and_format_agree'
    return False,reasons[0]

def localized_decision(p,job,result):
    """Require number ranges at the patch position before applying textual checks."""
    texts=[];px,py=p['xy'];cx,cy,cw,ch=job['context_box'];scale=CONFIG['context_scale'];pad=CONFIG['context_padding']
    near=[];has_number_at_position=False
    refh=max(20,(p['box'][3]-p['box'][1])*2)
    for line in result['lines']:
        if not line['candidates']:continue
        top=line['candidates'][0]
        if top['confidence']<.5:continue
        at_position=False
        for m in top['matches']:
            if canon(m['text'])!=canon(p['text']).strip('[]'):continue
            x,y,w,h=m['box'];l=cx+(x*job['context_width']-pad)/scale;t=cy+(y*job['context_height']-pad)/scale
            r=l+w*job['context_width']/scale;b=t+h*job['context_height']/scale
            if l-6<=px<=r+6 and t-6<=py<=b+6:at_position=True
        if at_position:has_number_at_position=True
        x,y,w,h=line['box'];source_y=cy+((y+h/2)*job['context_height']-pad)/scale
        if py-refh*.35<=source_y<=py+refh*.85:near.append((x,top['text']))
    if has_number_at_position and near:
        texts=[' '.join(t for x,t in sorted(near))]
    if not texts:return False,'no_matching_context_number_at_position',[]
    reasons=[]
    for text in texts:
        accept,reason=decide(p,text)
        if accept:return True,reason,texts
        reasons.append(reason)
    return False,reasons[0],texts
