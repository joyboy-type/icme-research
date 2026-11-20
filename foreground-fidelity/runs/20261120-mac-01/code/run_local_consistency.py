import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import json,time,hashlib
import torch,numpy as np,pandas as pd,timm
from timm.data import create_transform,resolve_model_data_config
from safetensors.torch import load_file
from PIL import Image
p=Path(__file__).parent
torch.set_num_threads(2);torch.set_num_interop_threads(2)
assert hashlib.sha256((p/'dinov2_vits14.safetensors').read_bytes()).hexdigest()=='04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081'
model=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=False,dynamic_img_size=True)
model.load_state_dict(load_file(str(p/'dinov2_vits14.safetensors')));model.eval()
data_config=resolve_model_data_config(model);data_config['input_size']=(3,224,224)
transform=create_transform(**data_config,is_training=False)
rows=json.loads((p/'fixed_sample_manifest.json').read_text())['rows'];av=json.loads((p/'availability_retry.json').read_text());ok={(x['idx'],x['part']) for x in av if x['ok']}
valid=[r for r in rows if all((r['row_idx'],part) in ok for part in ['source','A','B'])]
# Group identical source pixels before scoring, so one source never crosses splits.
source_group={}
for r in valid:
 with Image.open(p/'images'/f"{r['row_idx']:04d}_source.jpg") as im:
  source_group[r['row_idx']]=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
group_first={g:min(i for i,h in source_group.items() if h==g) for g in set(source_group.values())}
# No label used in encoding, saliency, scores, or rankings.
paths=[p/'images'/f"{r['row_idx']:04d}_{part}.jpg" for r in valid for part in ['source','A','B']]
features={};start=time.monotonic()
with torch.inference_mode():
 for i in range(0,len(paths),6):
  x=torch.stack([transform(Image.open(q).convert('RGB')) for q in paths[i:i+6]])
  ft=model.forward_features(x).float();ft=ft/(ft.norm(dim=-1,keepdim=True)+1e-12)
  for q,f in zip(paths[i:i+6],ft.numpy()):features[q.stem]=f
  if i%30==0:print('encoded',min(i+6,len(paths)),'/',len(paths),'seconds',round(time.monotonic()-start,1),flush=True)
np.savez_compressed(p/'dino_features.npz',keys=np.array(list(features)),features=np.array(list(features.values())))
score_rows=[]
rng=np.random.default_rng(20261120)
for r in valid:
 idx=r['row_idx'];src=features[f'{idx:04d}_source'];pat=src[1:];n=len(pat);k=n//4
 # Frozen source-only selector, shared by A and B. No difference mask or winner label used.
 sal=pat@src[0];sel=np.argsort(-sal,kind='stable')[:k]
 random=rng.choice(n,size=k,replace=False)
 grid=int(np.sqrt(n));ys,xs=np.divmod(np.arange(n),grid);center=np.argsort((ys-(grid-1)/2)**2+(xs-(grid-1)/2)**2,kind='stable')[:k]
 scores={}
 for part in ['A','B']:
  edt=features[f'{idx:04d}_{part}'];matrix=pat@edt[1:].T
  nearest=matrix.max(1)
  # All policies use the exact same input resolution, source and edited patch bank.
  scores[part]={'global_cls':float(src[0]@edt[0]),'all_patch_mean':float(nearest.mean()),'source_salient_quarter':float(nearest[sel].mean()),'random_quarter':float(nearest[random].mean()),'center_quarter':float(nearest[center].mean())}
 score_rows.append({'row_idx':idx,'key':r['key'],'scores':scores,'selected_source_patches':sel.tolist(),'source_group':source_group[idx],'split':'discovery' if group_first[source_group[idx]]<32 else 'holdout'})
(p/'scores_before_labels.json').write_text(json.dumps(score_rows,indent=2))
# Labels enter only below, for evaluation.
labels={r['row_idx']:r['winner'] for r in rows};results=[]
for r in score_rows:
 truth=labels[r['row_idx']];assert truth in ['Image A','Image B'];gt=truth[-1]
 out={'row_idx':r['row_idx'],'key':r['key'],'split':r['split'],'source_group':r['source_group'],'winner':gt}
 for m in r['scores']['A']:
  d=r['scores']['A'][m]-r['scores']['B'][m];pred='A' if d>0 else 'B' if d<0 else 'tie'
  out[m+'_A']=r['scores']['A'][m];out[m+'_B']=r['scores']['B'][m];out[m+'_pred']=pred;out[m+'_credit']=.5 if pred=='tie' else float(pred==gt)
 results.append(out)
df=pd.DataFrame(results);df.to_csv(p/'per_pair_results.csv',index=False)
methods=list(score_rows[0]['scores']['A'])
summary={'protocol':'Fixed first64 released background_change expert preference pairs. Nominal first32 discovery, next32 holdout; identical-source groups are assigned wholly to the first occurring split before scoring. No parameter fitting or policy change after scoring. Official DINOv2 ViT-S/14 feature backbone (not PVC-Judge or a reproduced SOTA VLM). All policies use the same single forward-pass token bank per source/candidate. Source-only 25% selector uses patch/CLS cosine; nearest match over candidate patches.','selected':len(rows),'executed':len(valid),'encoding_seconds':time.monotonic()-start,'splits':{}}
for split,g in [('all',df),('discovery',df[df.split=='discovery']),('holdout',df[df.split=='holdout'])]:
 out={'n':len(g),'source_groups':g.source_group.nunique(),'accuracy':{m:float(g[m+'_credit'].mean()) for m in methods},'paired':{}}
 for c in ['global_cls','all_patch_mean','random_quarter','center_quarter']:
  delta=(g.source_salient_quarter_credit-g[c+'_credit']).to_numpy();blocks=g.assign(delta=delta).groupby('source_group').delta.agg(['sum','size']).to_numpy();rng=np.random.default_rng(20261120);draws=[blocks[rng.integers(len(blocks),size=len(blocks))] for _ in range(2000)] if len(blocks) else [];bs=[v[:,0].sum()/v[:,1].sum() for v in draws] if draws else [0]
  out['paired']['salient_minus_'+c]={'delta':float(delta.mean()),'win':int((delta>0).sum()),'loss':int((delta<0).sum()),'tie':int((delta==0).sum()),'source_cluster_bootstrap_95ci':np.percentile(bs,[2.5,97.5]).tolist()}
 summary['splits'][split]=out
(p/'local_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
