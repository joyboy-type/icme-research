"""Bounded translation helpers for a development-only nuisance diagnostic."""
from collections import Counter
from functools import lru_cache
import cv2
import numpy as np

DEFAULTS={'max_integer_shift':3,'phase_min_response':.2,'tile_grid':[4,4],
          'tile_min_gray_std':3.,'min_voting_tiles':3,'min_vote_fraction':.6,'min_covered_quadrants':3}


@lru_cache(maxsize=32)
def window(shape):
    return cv2.createHanningWindow((shape[1],shape[0]),cv2.CV_32F)


def phase(a,b):
    aa=a.astype(np.float32);bb=b.astype(np.float32)
    aa-=aa.mean();bb-=bb.mean()
    (dx,dy),response=cv2.phaseCorrelate(aa,bb,window(aa.shape))
    if not np.isfinite([dx,dy,response]).all():return [0.,0.],0.
    return [float(dx),float(dy)],float(response)

def estimate_translation(before, after, mode, settings=None):
    if before.shape!=after.shape:raise ValueError('Image shapes differ')
    if mode not in ['none','global_phase','tile_consensus']:raise ValueError(mode)
    cfg={**DEFAULTS,**(settings or {})}
    out={'shift':[0,0],'response':0.,'raw_estimate':[0.,0.],'valid_tiles':0,
         'supporting_tiles':0,'covered_quadrants':0,'reason':'disabled'}
    if mode=='none':return out
    a,b=[cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in [before,after]]
    if min(float(a.std()),float(b.std()))<cfg['tile_min_gray_std']:
        out['reason']='low_texture';return out
    if mode=='global_phase':
        shift,response=phase(a,b);out.update(raw_estimate=shift,response=response)
        rounded=[int(np.rint(v)) for v in shift]
        if response<cfg['phase_min_response']:out['reason']='weak_phase_response'
        elif max(map(abs,rounded))>cfg['max_integer_shift']:out['reason']='outside_shift_bound'
        else:out.update(shift=rounded,reason='accepted' if any(rounded) else 'zero_estimate')
        return out
    votes=[];nr,nc=cfg['tile_grid'];h,w=a.shape
    for i in range(nr):
        for j in range(nc):
            rect=(slice(i*h//nr,(i+1)*h//nr),slice(j*w//nc,(j+1)*w//nc))
            pa,pb=a[rect],b[rect]
            if min(float(pa.std()),float(pb.std()))<cfg['tile_min_gray_std']:continue
            shift,response=phase(pa,pb);rounded=tuple(int(np.rint(v)) for v in shift)
            if response<cfg['phase_min_response'] or max(map(abs,rounded))>cfg['max_integer_shift']:continue
            quadrant=(int(i>=nr/2),int(j>=nc/2))
            votes.append((rounded,response,quadrant))
    out['valid_tiles']=len(votes)
    if not votes:out['reason']='no_valid_tiles';return out
    counts=Counter(v[0] for v in votes)
    best=sorted(counts,key=lambda x:(-counts[x],x))[0]
    support=[v for v in votes if v[0]==best];quadrants=len({v[2] for v in support})
    out.update(supporting_tiles=len(support),covered_quadrants=quadrants,response=float(np.mean([v[1] for v in support])),raw_estimate=list(best))
    if len(support)<cfg['min_voting_tiles']:out['reason']='insufficient_support'
    elif len(support)/len(votes)<cfg['min_vote_fraction']:out['reason']='no_consensus'
    elif quadrants<cfg['min_covered_quadrants']:out['reason']='localized_support'
    else:out.update(shift=list(best),reason='accepted' if any(best) else 'zero_estimate')
    return out

def translate(image, dx, dy):
    if dx==0 and dy==0:return image
    return cv2.warpAffine(image,np.float32([[1,0,dx],[0,1,dy]]),(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)

def residual_scores(before, after, margin=4):
    if before.shape!=after.shape:raise ValueError('Image shapes differ')
    if margin<0 or min(before.shape[:2])<=2*margin:raise ValueError('Invalid common margin')
    region=(slice(margin,-margin),slice(margin,-margin)) if margin else (slice(None),slice(None))
    delta=np.abs(before.astype(np.int16)-after.astype(np.int16)).max(axis=2)
    blurred=[cv2.GaussianBlur(x,(3,3),.6) for x in [before,after]]
    smooth=np.abs(blurred[0].astype(np.int16)-blurred[1].astype(np.int16)).max(axis=2)
    return {'pixel4':float((delta[region]>4).mean()),'blur8':float((smooth[region]>8).mean())}

def calibration_threshold(by_condition, max_fpr=.05):
    if not by_condition or not 0<=max_fpr<1:raise ValueError('Invalid calibration data')
    result=[]
    for values in by_condition.values():
        scores=np.asarray(values,float)
        if not len(scores) or not np.isfinite(scores).all():raise ValueError('Invalid scores')
        result.append(float(np.quantile(scores,1-max_fpr,method='higher')))
    return max(result)
