"""Prespecified geometry and content-only correspondence; no labels or tuning."""
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F,InterpolationMode
def spatial(image,mode,size=224,mask=False):
 if mode=='center':
  out=F.center_crop(F.resize(image,size,InterpolationMode.BICUBIC),[size,size])
  return out,np.ones((size,size),dtype=np.uint8)
 if mode=='square_warp':
  return image.resize((size,size),Image.Resampling.BICUBIC),np.ones((size,size),dtype=np.uint8)
 if mode!='letterbox':raise ValueError('unknown geometry')
 w,h=image.size;ratio=size/max(w,h);nw=max(1,round(w*ratio));nh=max(1,round(h*ratio));left=(size-nw)//2;top=(size-nh)//2
 fill=0. if mask else (124,116,104)
 out=Image.new('F' if mask else 'RGB',(size,size),fill);out.paste(image.resize((nw,nh),Image.Resampling.BICUBIC),(left,top))
 valid=np.zeros((size,size),dtype=np.uint8);valid[top:top+nh,left:left+nw]=1
 return out,valid
def patch_content(valid,patch=14):
 a=np.asarray(valid);h,w=a.shape
 if h%patch or w%patch:raise ValueError('content grid not divisible by patch size')
 coverage=a.reshape(h//patch,patch,w//patch,patch).mean((1,3)).reshape(-1)
 return np.flatnonzero(coverage>0)
def source_supports(prob,saliency,valid,seed):
 p=np.asarray(prob);sal=np.asarray(saliency);v=np.asarray(valid,dtype=int);side=int(np.sqrt(len(p)))
 if len(p)!=side*side or not len(v) or not np.isfinite(p).all():raise ValueError('invalid region grid')
 auto=v[p[v]>=.5]
 if not len(auto):auto=np.array([v[p[v].argmax()]])
 k=len(auto);y,x=np.divmod(v,side)
 return {'auto':auto,'salient':v[np.argsort(-sal[v],kind='stable')[:k]],'random':np.random.default_rng(seed).choice(v,k,replace=False),
         'center':v[np.argsort((y-(side-1)/2)**2+(x-(side-1)/2)**2,kind='stable')[:k]]}
def nearest(similarity,source_ids,candidate_ids,radius=None):
 m=np.asarray(similarity);src=np.asarray(source_ids,dtype=int);dst=np.asarray(candidate_ids,dtype=int)
 if not len(src) or not len(dst) or not np.isfinite(m).all():raise ValueError('empty/nonfinite correspondence input')
 chosen=m[np.ix_(src,dst)];missing=0
 if radius is not None:
  side=int(np.sqrt(m.shape[0]))
  if m.shape!=(side*side,side*side):raise ValueError('square full-grid coordinates required')
  distance=np.maximum(np.abs(src[:,None]//side-dst[None,:]//side),np.abs(src[:,None]%side-dst[None,:]%side))
  allowed=distance<=radius;missing=int((~allowed.any(axis=1)).sum())
  if missing:return {'score':None,'valid':False,'unmatched_source_patches':missing}
  chosen=np.where(allowed,chosen,-np.inf)
 return {'score':float(chosen.max(axis=1).mean()),'valid':True,'unmatched_source_patches':0}
