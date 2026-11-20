"""Isolation, control deduplication, metrics, and immutable file checks."""
from hashlib import sha256
import numpy as np

def unique_baselines(rows):
 result={}
 for r in rows:
  v={'id':r['baseline_id'],'family':r['family'],'split':r['split'],'before_path':r['before_path'],'before_sha256':r['before_sha256']}
  if v['id'] in result and result[v['id']]!=v:raise ValueError('Conflicting baseline identity')
  result[v['id']]=v
 return list(result.values())

def role_masks(rows,negative_conditions,positive_conditions):
 fit=np.array([r['split']=='train' and r['condition'] in (positive_conditions if r['positive'] else negative_conditions) for r in rows],bool)
 cal=np.array([r['split']=='development' and not r['positive'] and r['condition'] in negative_conditions for r in rows],bool)
 confirm=np.array([r['split']=='confirmation' for r in rows],bool)
 return fit,cal,confirm

def detection_metrics(rows,pred):
 result={'recall':{},'fpr':{},'per_class_clean_recall':{}}
 def rate(mask):
  p=np.asarray(pred,dtype=bool)[mask];return {'n':int(len(p)),'detected':int(p.sum()),'rate':float(p.mean())}
 for c in sorted({r['condition'] for r in rows}):
  for pos,key in [(1,'recall'),(0,'fpr')]:
   mask=np.array([r['condition']==c and r['positive']==pos for r in rows],bool)
   if mask.any():result[key][c]=rate(mask)
 for label in sorted({r['label'] for r in rows if r['positive']}):
  mask=np.array([r['positive']==1 and r['condition']=='clean' and r['label']==label for r in rows],bool)
  if mask.any():result['per_class_clean_recall'][label]=rate(mask)
 return result

def file_hashes(paths,root):return {str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
def check_lock(hashes,root):
 for relative,h in hashes.items():
  if sha256((root/relative).read_bytes()).hexdigest()!=h:raise ValueError('Locked file changed: '+relative)
