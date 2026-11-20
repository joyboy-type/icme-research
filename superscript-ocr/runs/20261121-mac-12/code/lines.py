"""Evaluation helpers for annotated physical line groups, never routing inputs."""
import re, unicodedata, difflib

def tokens(text):
    text=unicodedata.normalize('NFKC',text).lower()
    return re.findall(r'[a-z]+|[0-9]+',text)

def rectangle(poly):return [min(poly[::2]),min(poly[1::2]),max(poly[::2]),max(poly[1::2])]

def physical_lines(spans):
    groups=[]
    for span in sorted((s for s in spans if s['category_type']=='text_span'),key=lambda s:(rectangle(s['poly'])[1],rectangle(s['poly'])[0])):
        b=rectangle(span['poly']);cy=(b[1]+b[3])/2;h=b[3]-b[1]
        matches=[g for g in groups if abs(cy-g['cy'])<=.45*max(h,g['height'])]
        if matches:g=matches[-1];g['pieces'].append((b,span['text']))
        else:groups.append(dict(cy=cy,height=h,pieces=[(b,span['text'])]))
    out=[]
    for group in groups:
        pieces=sorted(group['pieces'],key=lambda p:p[0][0]);bs=[p[0] for p in pieces];text=' '.join(p[1] for p in pieces)
        if len(tokens(text))<4:continue
        out.append(dict(text=text,box=[min(b[0] for b in bs),min(b[1] for b in bs),max(b[2] for b in bs),max(b[3] for b in bs)],source_span_count=len(pieces)))
    return out

def covered_fraction(box,other):
    l,t,r,b=box;x,y,xx,yy=other
    return max(0,min(r,xx)-max(l,x))*max(0,min(b,yy)-max(t,y))/max(1,(r-l)*(b-t))

def recall(target,other):
    if not target:return 0
    return sum(m.size for m in difflib.SequenceMatcher(None,target,other,autojunk=False).get_matching_blocks())/len(target)

def compare_line(line,observations):
    target=tokens(line['text']);near=[o for o in observations if covered_fraction(line['box'],o['box'])>=.10]
    local=max((recall(target,tokens(o['text'])) for o in near),default=0)
    all_tokens=tokens(' '.join(o['text'] for o in observations))
    matcher=difflib.SequenceMatcher(None,target,all_tokens,autojunk=False)
    longest=matcher.find_longest_match().size
    global_recall=sum(m.size for m in matcher.get_matching_blocks())/max(1,len(target))
    present=local>=.65 or (longest>=min(4,len(target)) and global_recall>=.85)
    status='present' if present else 'omission_candidate' if not near else 'misread_or_merged_omission_candidate'
    return dict(screen_status=status,local_token_recall=local,global_token_recall=global_recall,longest_global_match=longest,geometric_observations=len(near),nearest_texts=[o['text'] for o in near])
