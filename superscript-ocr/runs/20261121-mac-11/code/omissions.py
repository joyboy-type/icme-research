"""Conservative candidate screening; deletion alignments are not ground truth."""
import re, html, unicodedata, difflib
from html.parser import HTMLParser


class Plain(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ['script','style']:self.skip+=1
        elif tag in ['p','div','br','tr','td','li','h1','h2','h3']:self.parts.append(' ')
    def handle_endtag(self,tag):
        if tag in ['script','style'] and self.skip:self.skip-=1
        else:self.parts.append(' ')
    def handle_data(self,text):
        if not self.skip:self.parts.append(text)


def tokens(raw):
    raw=raw.replace('Å',' angstrom ').replace('å',' angstrom ').replace('Å',' angstrom ')
    raw=re.sub(r'\\(?:overset|stackrel)\s*\{\s*\\circ\s*\}\s*\{\s*A\s*\}',' angstrom ',raw)
    raw=re.sub(r'\\AA\b',' angstrom ',raw)
    raw=re.sub(r'!\[[^\]]*\]\([^)]*\)',' ',raw)
    raw=re.sub(r'\[([^\]]+)\]\(https?://[^)]*\)',r'\1',raw)
    parser=Plain();parser.feed(raw);text=''.join(parser.parts)
    text=re.sub(r'\\[a-zA-Z]+',' ',text)
    return re.findall(r'[a-z]+|[0-9]+|[\u4e00-\u9fff]',unicodedata.normalize('NFKC',html.unescape(text)).lower())


def footnote_tokens(raw):
    # Space-separated glyphs inside one annotated footnote are not separate numbers.
    return tokens(re.sub(r'(?<=\d)\s+(?=\d)','',raw))


def anchored_gap(left,right,prediction):
    left_hits=[];right_hits=[]
    for n in range(min(6,len(left)),2,-1):
        target=left[-n:];hits=[i+n for i in range(len(prediction)-n+1) if prediction[i:i+n]==target]
        if len(hits)==1:left_hits=hits;break
    for n in range(min(6,len(right)),2,-1):
        target=right[:n];hits=[i for i in range(len(prediction)-n+1) if prediction[i:i+n]==target]
        if len(hits)==1:right_hits=hits;break
    pairs=[(l,r) for l in left_hits for r in right_hits if 0<=r-l<=12]
    return prediction[pairs[0][0]:pairs[0][1]] if len(pairs)==1 else None


def align(gold,prediction):
    return difflib.SequenceMatcher(None,gold,prediction,autojunk=False).get_opcodes()


def common_deletions(gold,predictions):
    shared=set(range(len(gold)))
    for prediction in predictions:
        deleted=set()
        for tag,a,b,c,d in align(gold,prediction):
            if tag=='delete':deleted.update(range(a,b))
        shared.intersection_update(deleted)
    runs=[]
    for i in sorted(shared):
        if runs and i==runs[-1][1]:runs[-1]=(runs[-1][0],i+1)
        else:runs.append((i,i+1))
    return runs


def normal_looking(prediction):
    if not prediction:return False
    ngrams=[tuple(prediction[i:i+10]) for i in range(max(0,len(prediction)-9))]
    return not ngrams or len(set(ngrams))/len(ngrams)>=.5
