"""Original-byte and pixel checks for source-verified version screenshot pairs."""
from hashlib import sha1,sha256
import numpy as np

def git_blob_sha(data):return sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
def pixel_audit(a,b):
 if a.ndim!=3 or b.ndim!=3 or a.shape[2]!=3 or b.shape[2]!=3:raise ValueError('Expected RGB images')
 result={'before_shape':list(a.shape),'after_shape':list(b.shape),'same_geometry':a.shape==b.shape,'before_pixel_sha256':sha256(a.tobytes()).hexdigest(),'after_pixel_sha256':sha256(b.tobytes()).hexdigest()}
 h,w=min(a.shape[0],b.shape[0]),min(a.shape[1],b.shape[1]);d=np.abs(a[:h,:w].astype(np.int16)-b[:h,:w].astype(np.int16)).max(2);yy,xx=np.nonzero(d)
 result.update(overlap_changed_pixels=int((d>0).sum()),overlap_changed_gt4=int((d>4).sum()),overlap_shape=[h,w],overlap_max_delta=int(d.max()),changed_pixels=None,changed_gt4=None,changed_fraction=None,bbox_xyxy=None)
 if result['same_geometry']:
  result.update(changed_pixels=int((d>0).sum()),changed_gt4=int((d>4).sum()),changed_fraction=float((d>0).mean()),bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None)
 return result
