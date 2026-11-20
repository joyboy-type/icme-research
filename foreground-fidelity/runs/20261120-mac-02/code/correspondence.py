"""Frozen diagnostic rules; selection depends on source data only."""
import numpy as np

def mean_nearest(similarity, selected, radius=None):
    matrix=np.asarray(similarity)
    ids=np.asarray(selected,dtype=int)
    if not len(ids) or not np.isfinite(matrix).all():
        raise ValueError('empty support or nonfinite features')
    chosen=matrix[ids]
    if radius is not None:
        side=int(np.sqrt(matrix.shape[0]))
        if matrix.shape!=(side*side,side*side):
            raise ValueError('equal square patch grids required')
        allids=np.arange(side*side)
        allowed=np.maximum(np.abs(ids[:,None]//side-allids[None,:]//side),
                           np.abs(ids[:,None]%side-allids[None,:]%side))<=radius
        chosen=np.where(allowed,chosen,-np.inf)
    return float(chosen.max(axis=1).mean())

def supports(mask_probabilities, source_tokens, seed):
    probs=np.asarray(mask_probabilities)
    if not np.isfinite(probs).all():raise ValueError('nonfinite mask')
    n=len(probs);grid=int(np.sqrt(n))
    if n!=grid*grid:raise ValueError('square source grid required')
    subject=np.flatnonzero(probs>=.5)
    if not len(subject):subject=np.array([int(probs.argmax())])
    k=len(subject)
    sal=source_tokens[1:]@source_tokens[0]
    y,x=np.divmod(np.arange(n),grid)
    return {'subject':subject,
            'salient':np.argsort(-sal,kind='stable')[:k],
            'random':np.random.default_rng(seed).choice(n,k,replace=False),
            'center':np.argsort((y-(grid-1)/2)**2+(x-(grid-1)/2)**2,kind='stable')[:k]}
