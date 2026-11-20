"""Only reserved manifest and original source images are read; no preference data."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import json,hashlib,math
import numpy as np,pandas as pd,cv2
from PIL import Image,ImageDraw
cv2.setNumThreads(2)
R=Path(__file__).resolve().parents[1];root=R.parents[1];sha=lambda b:hashlib.sha256(b).hexdigest()
reservation=root/'data/snapshot-03-confirmation-reserved/manifest.json';manifest=json.loads(reservation.read_text())
assert manifest['source_groups']==20 and manifest['pairs']==37 and manifest['new_predictions']==0
old=json.loads((root/'runs/20261120-mac-02/outputs/input_audit_before_scoring.json').read_text())
development={}
for r in old:development.setdefault(r['source_group'],{'source_group':r['source_group'],'source_file':r['images']['source'],'first_row':r['row_idx']})
reserved=[{'source_group':g['pixel_hash'],'source_file':g['source_file'],'review_index':i,'pair_count':len(g['pairs']),'row_indices':[r['row_idx'] for r in g['pairs']]} for i,g in enumerate(manifest['groups'])]
assert len(development)==46 and len(reserved)==20 and sum(x['pair_count'] for x in reserved)==37
def phash(im):
 x=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32);v=cv2.dct(x)[:8,:8].ravel()[1:];return v>np.median(v)
def features(r):
 p=Path(r['source_file']);im=Image.open(p).convert('RGB');w,h=im.size;dx=.1*w;dy=.1*h;crop=im.crop((round(dx),round(dy),round(w-dx),round(h-dy)))
 gray=np.asarray(im.convert('L').resize((9,8),Image.Resampling.LANCZOS))
 return {**r,'wh':[w,h],'file_sha256':sha(p.read_bytes()),'RGB_sha256':sha(im.tobytes()),'canonical_sha256':sha(str(im.size).encode()+im.tobytes()),'phash':[phash(im),phash(crop)],'dhash':(gray[:,1:]>gray[:,:-1]).ravel()}
dev=[features(r) for r in development.values()];new=[features(r) for r in reserved]
for r in dev+new:assert r['RGB_sha256']==r['source_group']
def distance(a,b,kind):
 return {'left':a['source_group'],'right':b['source_group'],'kind':kind,'exact_RGB':a['RGB_sha256']==b['RGB_sha256'],'exact_canonical':a['canonical_sha256']==b['canonical_sha256'],'exact_file':a['file_sha256']==b['file_sha256'],'phash_full':int(np.count_nonzero(a['phash'][0]!=b['phash'][0])),'phash_any_full_crop':min(int(np.count_nonzero(x!=y)) for x in a['phash'] for y in b['phash']),'dhash_full':int(np.count_nonzero(a['dhash']!=b['dhash']))}
cross=[distance(a,b,'reserved_vs_development') for a in new for b in dev];within=[distance(a,b,'within_reserved') for i,a in enumerate(new) for b in new[i+1:]]
allpairs=cross+within
for p in allpairs:p['flag']=p['exact_RGB'] or p['phash_any_full_crop']<=12 or p['dhash_full']<=10
pd.DataFrame(cross).to_csv(R/'outputs/cross_source_hash_distances.csv',index=False);pd.DataFrame(within).to_csv(R/'outputs/within_reserved_hash_distances.csv',index=False)
flags=[p for p in allpairs if p['flag']];(R/'outputs/flagged_source_pairs.json').write_text(json.dumps(flags,indent=2))
lookup={r['source_group']:r for r in dev+new};reviews=[]
for a in new:
 candidates=[p for p in cross if p['left']==a['source_group']];picks=[]
 for metric in ['phash_full','phash_any_full_crop','dhash_full']:
  p=min(candidates,key=lambda p:(p[metric],p['right']))
  if p['right'] not in picks:picks.append(p['right'])
 for p in sorted(candidates,key=lambda p:(p['phash_full'],p['right'])):
  if len(picks)>=3:break
  if p['right'] not in picks:picks.append(p['right'])
 reviews.append({'review_index':a['review_index'],'source_group':a['source_group'],'pair_count':a['pair_count'],'row_indices':a['row_indices'],'nearest_candidates':[{**next(p for p in candidates if p['right']==b),'development_first_row':lookup[b]['first_row']} for b in picks]})
for page in range(math.ceil(len(reviews)/3)):
 sheet=Image.new('RGB',(1040,840),'white');d=ImageDraw.Draw(sheet)
 for j,a in enumerate(reviews[page*3:(page+1)*3]):
  y=j*280;items=[(a['source_group'],f"reserved {a['review_index']:02d} rows {a['row_indices']}")]+[(p['right'],f"dev row{p['development_first_row']} p{p['phash_full']} c{p['phash_any_full_crop']} d{p['dhash_full']}") for p in a['nearest_candidates']]
  for k,(g,title) in enumerate(items):
   im=Image.open(lookup[g]['source_file']).convert('RGB');im.thumbnail((250,240));sheet.paste(im,(k*260+2,y+28));d.text((k*260+2,y+4),title,fill='black')
 sheet.save(R/f'qa/nearest_source_review_{page+1}.png')
for page in range(math.ceil(len(flags)/4)):
 sheet=Image.new('RGB',(800,1120),'white');d=ImageDraw.Draw(sheet)
 for j,p in enumerate(flags[page*4:(page+1)*4]):
  for k,key in enumerate(['left','right']):
   r=lookup[p[key]];im=Image.open(r['source_file']).convert('RGB');im.thumbnail((390,240));sheet.paste(im,(k*400+2,j*280+32));d.text((k*400+2,j*280+4),f"{p['kind']} {key} {p[key][:10]} p{p['phash_full']} c{p['phash_any_full_crop']} d{p['dhash_full']}",fill='black')
 sheet.save(R/f'qa/flagged_pair_review_{page+1}.png')
public_inventory=[{k:v for k,v in r.items() if k not in ['source_file','phash','dhash']} for r in dev+new]
(R/'outputs/source_inventory_screened.json').write_text(json.dumps(public_inventory,indent=2));(R/'outputs/source_review_manifest.json').write_text(json.dumps(reviews,indent=2))
summary={'reserved_groups':20,'reserved_pairs':37,'development_groups':46,'development_pairs':89,'cross_comparisons':len(cross),'within_comparisons':len(within),'exact_cross_RGB_matches':sum(p['exact_RGB'] for p in cross),'exact_within_RGB_matches':sum(p['exact_RGB'] for p in within),'flagged_cross_pairs':sum(p['flag'] for p in cross),'flagged_within_pairs':sum(p['flag'] for p in within),'status':'awaiting_source_visual_review','confirmation_preference_labels_opened':False,'confirmation_AB_images_opened':False,'preference_model_or_scores_run':False,'manifest_sha256':sha(reservation.read_bytes()),'frozen_protocol_sha256':sha((R/'protocol/audit_frozen.json').read_bytes())}
(R/'outputs/source_audit_summary_before_review.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
