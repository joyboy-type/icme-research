"""Pixel-OCR-only citation parsing and a deliberately limited consistency rule."""
import re
import unicodedata


def words(text):
    return re.findall(r'[a-z]{2,}', unicodedata.normalize('NFKC', text).lower())


def expand(text):
    result = []
    for part in re.split(r'[,;]', text.replace(' ', '')):
        ends = re.split(r'[-–—−]', part)
        if len(ends) == 1 and ends[0].isdigit():
            result.append(int(ends[0]))
        elif len(ends) == 2 and all(x.isdigit() for x in ends):
            lo, hi = map(int, ends)
            if not 0 < lo <= hi <= 999 or hi-lo > 50:
                return []
            result.extend(range(lo, hi+1))
        else:
            return []
    return sorted(set(result)) if all(0 < n < 1000 for n in result) else []


def citations(text):
    out = []
    for match in re.finditer(r'\[\s*([0-9][0-9\s,;–—−-]*)\s*\]', text):
        numbers = expand(match[1])
        if numbers:
            out.append(dict(numbers=numbers, start=match.start(), end=match.end(),
                            left=text[max(0,match.start()-90):match.start()],
                            right=text[match.end():match.end()+90]))
    return out


def admit(candidate, bibliography, before, after, check_anchor):
    if not all(n in bibliography for n in candidate['numbers']):
        return False
    if not check_anchor:
        return True
    pairs = [(words(candidate['left'])[-3:], words(before)[-3:]),
             (words(candidate['right'])[:3], words(after)[:3])]
    checked = 0
    for one, two in pairs:
        if one and two:
            checked += 1
            if len(set(one) & set(two)) < min(2, len(set(one)), len(set(two))):
                return False
    return checked > 0


def read_bibliography(observations):
    result={};started=False;current=None
    for page_id,text in observations:
        if re.fullmatch(r'\s*References\s*',text,re.I):
            started=True;continue
        if not started:continue
        match=re.match(r'^\s*([1-9][0-9]{0,2})[.)](?:\s+(.*)|\s*)$',text)
        if match:
            current=int(match[1]);result[current]=dict(text=match[2] or '',page_id=page_id)
        elif current is not None:
            result[current]['text']=(result[current]['text']+' '+text).strip()
    return result
