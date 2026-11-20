"""Raster-only proposals and conservative attachment; no label file access."""
import math
import unicodedata
import re
import cv2
import numpy as np

CONFIG = dict(max_crops=32, scale=4, padding=32, component_min_height=.20,
              component_max_height=.80, elevation=.16, group_gap=.52,
              confidence=.50, crop_margin=2)
cv2.setNumThreads(2)


def numeric(s):
    return unicodedata.normalize('NFKC', str(s)).translate(str.maketrans('–−—', '---')).strip().strip('[]')


def proposals(gray, lines):
    height, width = gray.shape
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 12)
    _, _, boxes, _ = cv2.connectedComponentsWithStats(mask)
    boxes = boxes[1:]
    xx, yy, ww, hh, aa = boxes.T
    out = []
    for li, line in enumerate(lines):
        if len(line['text']) < 14:
            continue
        x, y, w, h = np.array(line['box']) * [width, height, width, height]
        if not 8 <= h <= height*.04:
            continue
        near = (xx >= x-h*.25) & (xx+ww <= x+w+h) & (yy >= y-h*.55) & (yy+hh <= y+h*1.20)
        body = near & (hh >= h*.50) & (hh <= h*1.1) & (ww <= h*1.8) & (aa >= 8)
        if body.sum() < 5:
            continue
        baseline = float(np.median((yy+hh)[body]))
        valid = (near & (hh >= max(3, h*CONFIG['component_min_height'])) &
                 (hh <= h*CONFIG['component_max_height']) & (ww >= 2) &
                 (ww <= h*1.2) & (aa >= 5) &
                 (yy+hh <= baseline-h*CONFIG['elevation']) & (yy <= y+h*.42))
        pieces = sorted(boxes[valid].tolist())
        groups = []
        for bx, by, bw, bh, area in pieces:
            if groups and bx-groups[-1][2] <= h*CONFIG['group_gap'] and abs((by+bh/2)-(groups[-1][1]+groups[-1][3])/2) <= h*.30:
                g = groups[-1]
                g[:] = [g[0], min(by,g[1]), max(bx+bw,g[2]), max(by+bh,g[3]), g[4]+area]
            else:
                groups.append([bx,by,bx+bw,by+bh,area])
        for l,t,r,b,area in groups:
            if r-l > h*4 or b-t < h*.20:
                continue
            # Require a nearby baseline character to the left; isolated figure labels abstain.
            left = body & (xx+ww <= l+h*.15) & (xx+ww >= l-h*1.8)
            if not left.any():
                continue
            score = (baseline-b)/h + min((r-l)/h, 1.5)*.15 + min(area/(h*h), .5)*.1
            margin = CONFIG['crop_margin']
            box = [max(0,l-margin),max(0,t-margin),min(width,r+margin),min(height,b+margin)]
            out.append(dict(box=box,line_id=li,score=score,baseline=baseline,line_height=float(h)))
    out.sort(key=lambda p:(-p['score'],p['box'][1],p['box'][0]))
    unique = []
    for p in out:
        l,t,r,b=p['box']
        if any(max(l,q['box'][0]) < min(r,q['box'][2]) and max(t,q['box'][1]) < min(b,q['box'][3]) for q in unique):
            continue
        unique.append(p)
        if len(unique) == CONFIG['max_crops']:
            break
    return unique


def halton(i, base):
    value, f = 0., 1.
    while i:
        f /= base
        value += f*(i % base)
        i //= base
    return value


def paired_jobs(props, width, height):
    jobs = []
    for i, p in enumerate(props):
        l,t,r,b = p['box']; w,h = r-l,b-t
        ux = round(halton(i+1,2)*(width-w)); uy = round(halton(i+1,3)*(height-h))
        jobs.append(dict(method='candidate', index=i, **p))
        jobs.append(dict(method='uniform', index=i, box=[ux,uy,ux+w,uy+h]))
    return jobs


def attach(lines, predictions, base_numbers):
    patches = []
    for p in predictions:
        identity = numeric(p['text'])
        if not re.fullmatch(r'[0-9]{1,3}(?:[,\-][0-9]{1,3})*', identity):
            continue
        px,py = p['xy']; pl,pt,pr,pb = p['box']; ph = max(3,pb-pt)
        # Abstain both for duplicates and conflicts with already present numeric tokens.
        if any(abs(q['xy'][0]-px) <= max(6,pr-pl) and abs(q['xy'][1]-py) <= max(6,ph)
               for q in base_numbers):
            continue
        choices=[]
        for i,line in enumerate(lines):
            x,y,w,h=line['box']
            if x-h*.2 <= px <= x+w+h and y-h*.65 <= py <= y+h*.55 and ph <= h*.9:
                choices.append((abs(py-y)/h,i))
        if not choices:
            continue
        _, li = min(choices)
        line=lines[li]
        chars=[c for c in line['chars'] if c['text'].strip() and c['box'][0]+c['box'][2] <= px+1]
        if not chars:
            continue
        c=max(chars,key=lambda c:c['box'][0]+c['box'][2]); offset=c['end']
        if px-(c['box'][0]+c['box'][2]) > line['box'][3]*1.8:
            continue
        if any(q['line_id']==li and q['offset']==offset for q in patches):
            continue
        patches.append(dict(**p,identity=identity,line_id=li,offset=offset,
                            left=line['text'][:offset],right=line['text'][offset:]))
    return patches
