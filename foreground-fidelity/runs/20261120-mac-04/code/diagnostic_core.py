import numpy as np
def reviewed_support(boxes,eligible,side=16):
 if not eligible:return np.arange(side*side)
 if not boxes:raise ValueError('eligible subject needs a nonempty box union')
 y,x=np.divmod(np.arange(side*side),side);selected=np.zeros(side*side,dtype=bool)
 for x0,y0,x1,y1 in boxes:
  if not (0<=x0<x1<=side and 0<=y0<y1<=side):raise ValueError('box outside crop grid')
  selected|=(x+.5>=x0)&(x+.5<x1)&(y+.5>=y0)&(y+.5<y1)
 ids=np.flatnonzero(selected)
 if not len(ids):raise ValueError('review box selects no patch center')
 return ids

def matched_controls(subject,probabilities,saliency,seed):
 prob=np.asarray(probabilities);sal=np.asarray(saliency);n=len(prob);side=int(np.sqrt(n));k=len(subject)
 if prob.shape!=sal.shape or n!=side*side or not 0<k<=n:raise ValueError('invalid control area/grid')
 if not np.isfinite(prob).all() or not np.isfinite(sal).all():raise ValueError('nonfinite control score')
 y,x=np.divmod(np.arange(n),side)
 return {'reviewed':np.array(subject,dtype=int),'u2_topk':np.argsort(-prob,kind='stable')[:k],
         'salient_topk':np.argsort(-sal,kind='stable')[:k],
         'random_topk':np.random.default_rng(seed).choice(n,k,replace=False),
         'center_topk':np.argsort((y-(side-1)/2)**2+(x-(side-1)/2)**2,kind='stable')[:k]}

def match_diagnostic(similarity,selected,radius=2):
 m=np.asarray(similarity);ids=np.asarray(selected,dtype=int);side=int(np.sqrt(m.shape[0]))
 if m.shape!=(side*side,side*side) or not len(ids) or not np.isfinite(m).all():raise ValueError('invalid correspondence matrix/support')
 allids=np.arange(side*side)
 dist=np.maximum(np.abs(ids[:,None]//side-allids[None,:]//side),np.abs(ids[:,None]%side-allids[None,:]%side))
 chosen=m[ids];best=chosen.argmax(axis=1);displacement=dist[np.arange(len(ids)),best]
 free=float(chosen.max(axis=1).mean());restricted=float(np.where(dist<=radius,chosen,-np.inf).max(axis=1).mean())
 return {'outside_radius_fraction':float((displacement>radius).mean()),'best_match_distance_median':float(np.median(displacement)),
         'best_match_distance_p95':float(np.percentile(displacement,95)),'score_unrestricted':free,'score_restricted':restricted,'score_penalty':free-restricted}
