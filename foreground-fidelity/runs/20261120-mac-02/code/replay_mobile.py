import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,time,hashlib
import torch,numpy as np,pandas as pd,open_clip
from PIL import Image
R=Path(__file__).resolve().parents[1];BASE=R.parents[1];OUT=R/'outputs'
torch.set_num_threads(2);torch.set_num_interop_threads(2)
weight=BASE/'data/snapshot-02/weights/mobileclip2_s0.pt'
assert hashlib.sha256(weight.read_bytes()).hexdigest()=='bc3bc861baa680df2f9e3dc1aa0acbe5a21af21032df385026dda70c33505aa6'
model,_,pre=open_clip.create_model_and_transforms('MobileCLIP2-S0',pretrained=str(weight),image_mean=(0,0,0),image_std=(1,1,1));model.eval()
rows=json.loads((R/'protocol/inputs.json').read_text())['rows'];labels={r['pair_id']:r.pop('expert_winner')[-1] for r in rows};ft={};times={};started=time.perf_counter()
items={filename:None for r in rows for filename in r['images'].values()};paths=list(items)
with torch.inference_mode():
 for i in range(0,len(paths),8):
  batch=paths[i:i+8];x=torch.stack([pre(Image.open(f).convert('RGB')) for f in batch]);t=time.perf_counter();y=model.encode_image(x).float();y=y/y.norm(dim=-1,keepdim=True);elapsed=time.perf_counter()-t
  for f,v in zip(batch,y.numpy()):ft[f]=v;times[f]=elapsed/len(batch)
records=[]
for r in rows:
 src=ft[r['images']['source']];aa=float(src@ft[r['images']['A']]);bb=float(src@ft[r['images']['B']]);records.append({'pair_id':r['pair_id'],'score_a':aa,'score_b':bb,'winner':'A' if aa>bb else 'B' if bb>aa else 'tie','amortized_ms':sum(times[r['images'][k]] for k in ['source','A','B'])*1000})
(OUT/'mobile_scores_before_labels.json').write_text(json.dumps(records,indent=2))
df=pd.DataFrame(records);df['expert_winner']=df.pair_id.map(labels);df['credit']=np.where(df.winner=='tie',.5,(df.winner==df.expert_winner).astype(float));df.to_csv(OUT/'mobile_per_pair.csv',index=False)
assert int(df.credit.sum())==72
summary={'pairs':len(df),'correct':int(df.credit.sum()),'ties':int((df.winner=='tie').sum()),'accuracy':float(df.credit.mean()),'fresh_replay_matches_72_of_89':True,'time_seconds':time.perf_counter()-started,'forward_seconds':sum(times.values()),'device':'cpu','threads':2,'open_clip_version':open_clip.__version__}
(OUT/'mobile_replay.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
