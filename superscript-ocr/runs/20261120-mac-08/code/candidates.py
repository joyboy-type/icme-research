"""Geometry-only reranking and component isolation. No labels or OCR numbers used."""
import math
import cv2
import numpy as np

CONFIG=dict(max_crops=32,scale=4,canvas_padding=32,component_area_min=5,
            primary_height_fraction=.55,minimum_primary_line_fraction=.25,
            retained_edge_margin=1,antialias_dilation=1,
            height_target=.48,height_sigma=.20,elevation_target=.28,
            elevation_sigma=.18,nms_overlap=.50)
cv2.setNumThreads(2)

def isolate(proposal,cc,stats):
    l,t,r,b=proposal['box'];lineh=proposal['line_height']
    selected=[(i,s) for i,s in enumerate(stats) if i and s[4]>=CONFIG['component_area_min'] and
              l<=s[0] and t<=s[1] and s[0]+s[2]<=r and s[1]+s[3]<=b]
    if not selected:return None
    tallest=max(s[3] for i,s in selected)
    primary=[(i,s) for i,s in selected if s[3]>=tallest*CONFIG['primary_height_fraction']]
    if tallest<lineh*CONFIG['minimum_primary_line_fraction']:return None
    x1=min(s[0] for i,s in primary);y1=min(s[1] for i,s in primary)
    x2=max(s[0]+s[2] for i,s in primary);y2=max(s[1]+s[3] for i,s in primary)
    # Preserve small separators only between primary glyphs, not leading body fragments.
    small=[(i,s) for i,s in enumerate(stats) if i and s[4]>=2 and s[3]<tallest*CONFIG['primary_height_fraction'] and
           x1<=s[0] and s[0]+s[2]<=x2 and y1<=s[1] and s[1]+s[3]<=y2+2]
    kept=primary+small
    y2=max(s[1]+s[3] for i,s in kept)
    margin=CONFIG['retained_edge_margin'];height,width=cc.shape
    box=[max(0,int(x1)-margin),max(0,int(y1)-margin),min(width,int(x2)+margin),min(height,int(y2)+margin)]
    glyphh=max(s[3] for i,s in primary);ratio=glyphh/lineh
    elevation=(proposal['baseline']-max(s[1]+s[3] for i,s in primary))/lineh
    heightfit=math.exp(-.5*((ratio-CONFIG['height_target'])/CONFIG['height_sigma'])**2)
    elevfit=math.exp(-.5*((elevation-CONFIG['elevation_target'])/CONFIG['elevation_sigma'])**2)
    consistency=min(s[3] for i,s in primary)/glyphh
    score=2*heightfit+elevfit+.3*consistency+.2*min(len(primary),3)/3
    return dict(box=box,raw_box=proposal['box'],component_ids=[int(i) for i,s in kept],
                score=float(score),old_score=proposal['score'],line_id=proposal['line_id'],
                line_height=lineh,baseline=proposal['baseline'],primary_height=int(glyphh),
                primary_count=len(primary),height_ratio=float(ratio),elevation=float(elevation))

def rerank(pool,cc,stats,limit=32):
    ranked=[c for p in pool if (c:=isolate(p,cc,stats)) is not None]
    ranked.sort(key=lambda c:(-c['score'],c['box'][1],c['box'][0]))
    out=[]
    for c in ranked:
        l,t,r,b=c['box'];area=(r-l)*(b-t)
        if any(max(0,min(r,q['box'][2])-max(l,q['box'][0]))*
               max(0,min(b,q['box'][3])-max(t,q['box'][1]))/min(area,(q['box'][2]-q['box'][0])*(q['box'][3]-q['box'][1]))>CONFIG['nms_overlap'] for q in out):continue
        out.append(c)
        if len(out)>=limit:break
    return out

def render(gray,cc,clean):
    l,t,r,b=clean['box'];crop=gray[t:b,l:r].copy()
    mask=np.isin(cc[t:b,l:r],clean['component_ids']).astype('uint8')
    radius=CONFIG['antialias_dilation']
    if radius:mask=cv2.dilate(mask,np.ones((radius*2+1,radius*2+1),np.uint8))
    crop[mask==0]=255
    return crop
