"""Pixel-only helpers for the locked local web-change experiment."""
import cv2
import numpy as np
from legacy_paired import projection_stats

def crop_xyxy(before, after):
    if before.shape != after.shape:
        raise ValueError('Screenshot shapes differ')
    d=np.abs(before.astype(np.int16)-after.astype(np.int16)).max(axis=2)
    yy,xx=np.nonzero(d>(4 if (d>4).any() else 0))
    if not len(xx): return None
    pad=max(4,round(.12*min(np.ptp(yy)+1,np.ptp(xx)+1)))
    return (int(max(0,xx.min()-pad)),int(max(0,yy.min()-pad)),
            int(min(before.shape[1],xx.max()+1+pad)),int(min(before.shape[0],yy.max()+1+pad)))

def edge_blocks(before_edges, after_edges, mode):
    a,b=before_edges,after_edges
    new=b&~a; old=a&~b
    def directional(x,y):
        n=y&~x; o=x&~y
        return np.r_[[x.mean(),y.mean(),n.mean(),o.mean(),(x&y).sum()/max(1,(x|y).sum())],
                     projection_stats(n.astype(np.float32)),projection_stats(o.astype(np.float32))]
    if mode=='directional': return directional(a,b)
    if mode=='symmetric': return .5*(directional(a,b)+directional(b,a))
    if mode=='new_only': return directional(np.zeros_like(a),new)
    if mode=='old_only': return directional(old,np.zeros_like(b))
    if mode=='merged':
        changed=new|old
        common=[.5*(a.mean()+b.mean())]*2+[.5*changed.mean()]*2+[(a&b).sum()/max(1,(a|b).sum())]
        return np.r_[common,projection_stats(changed.astype(np.float32))*2]
    raise ValueError(mode)

def connected_groups(ids, edges):
    parent=list(range(len(ids)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,j in edges:
        ri,rj=root(i),root(j)
        if ri!=rj: parent[rj]=ri
    names={}
    for i,name in enumerate(ids): names[root(i)]=min(name,names.get(root(i),name))
    return [names[root(i)] for i in range(len(ids))]

def false_positive_threshold(scores, max_fpr=.05):
    return float(np.quantile(scores,1-max_fpr,method='higher'))
