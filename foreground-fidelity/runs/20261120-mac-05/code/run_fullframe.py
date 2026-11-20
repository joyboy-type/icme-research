"""Fresh full-frame diagnostics on old pairs only. Word and confirmation untouched."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import sys,argparse,json,hashlib,time,platform
import numpy as np,pandas as pd,torch,timm
from PIL import Image
from skimage import transform as sktransform
from safetensors.torch import load_file
from timm.data import resolve_model_data_config
from torchvision.transforms import functional as F
from fullframe_core import spatial,patch_content,source_supports,nearest
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--archive-root',type=Path,default=R.parents[1]);a=p.parse_args();root=a.archive_root.resolve();old=root/'runs/20261120-mac-02';out=R/'outputs'
sys.path.insert(0,str(old/'code'));from u2net_vendor import U2NETP
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(2);torch.set_num_interop_threads(2)
modes=['center','letterbox','square_warp'];audit=load(old/'outputs/input_audit_before_scoring.json');source_meta=load(old/'outputs/source_mask_metadata_before_labels.json')
assert len(audit)==89 and len(source_meta)==46
canonical={}
for row in audit:
 for part,filename in row['images'].items():
  im=Image.open(filename).convert('RGB');h=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest();assert h==row['pixel_hashes'][part]
  canonical.setdefault(h,filename)
dino_weight=root/'data/snapshot-01/weights/dinov2_vits14.safetensors';u2_weight=root/'data/snapshot-02/weights/u2netp.pth'
assert sha(dino_weight)=='04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081'
assert sha(u2_weight)=='e7567cde013fb64813973ce6e1ecc25a80c05c3ca7adbc5a54f3c3d90991b854'
word=Path(os.environ['FROZEN_TASKBOOK_PATH']) if os.environ.get('FROZEN_TASKBOOK_PATH') else None
input_seal={'method_sha256':sha(R/'protocol/method_frozen.json'),'audit_sha256':sha(old/'outputs/input_audit_before_scoring.json'),'prior_source_metadata_sha256':sha(old/'outputs/source_mask_metadata_before_labels.json'),'prior_feature_sha256':sha(old/'outputs/dino_features.npz'),'dino_weight_sha256':sha(dino_weight),'U2NETP_weight_sha256':sha(u2_weight),'Word_sha256_before':sha(word) if word is not None else None,'pairs':89,'source_groups':46,'unique_images':len(canonical),'stage':'before_new_inference'}
(R/'protocol/input_seal.json').write_text(json.dumps(input_seal,indent=2))
model=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=False,dynamic_img_size=True);model.load_state_dict(load_file(str(dino_weight)));model.eval()
config=resolve_model_data_config(model);mean=config['mean'];std=config['std'];features={};content={};forward_time={};encode_begin=time.perf_counter()
items=list(canonical.items())
with torch.inference_mode():
 for mode in modes:
  features[mode]={};content[mode]={};total=0.
  for start in range(0,len(items),6):
   batch=items[start:start+6];images=[]
   for h,path in batch:
    im=Image.open(path).convert('RGB');prepared,valid=spatial(im,mode)
    content[mode][h]={'original_wh':im.size,'valid_patch_ids':patch_content(valid).tolist(),'content_pixel_fraction':float(np.asarray(valid).mean())}
    images.append(F.normalize(F.to_tensor(prepared),mean,std))
   x=torch.stack(images);t=time.perf_counter();f=model.forward_features(x).float();f=f/(f.norm(dim=-1,keepdim=True)+1e-12);total+=time.perf_counter()-t
   for (h,_),tokens in zip(batch,f.numpy()):features[mode][h]=tokens
  forward_time[mode]=total
  np.savez_compressed(out/f'dino_features_{mode}.npz',keys=np.array(list(features[mode])),features=np.array(list(features[mode].values())))
  print('encoded',mode,len(features[mode]),'unique images; model forward seconds',round(total,3),flush=True)
 del model
oldf=np.load(old/'outputs/dino_features.npz');oldfeatures=dict(zip(oldf['keys'],oldf['features']))
maxerr=max(float(np.abs(features['center'][h]-oldfeatures[h]).max()) for h in oldfeatures)
assert maxerr<1e-5,('center feature replay drift',maxerr)
(out/'spatial_metadata_before_scoring.json').write_text(json.dumps(content,indent=2))
seg=U2NETP(3,1);seg.load_state_dict(torch.load(u2_weight,map_location='cpu',weights_only=True));seg.eval()
supports={mode:{} for mode in modes};mask_vectors={mode:{} for mode in modes};mask_records=[];raw320={};t_seg=time.perf_counter()
with torch.inference_mode():
 for index,m in enumerate(source_meta):
  g=m['source_group'];h=m['source_pixel_sha'];im=Image.open(m['source_file']).convert('RGB');array=np.asarray(im)
  x=sktransform.resize(array,(320,320),mode='constant');x=x/max(float(x.max()),1e-12);x=(x-np.array([.485,.456,.406]))/np.array([.229,.224,.225]);x=torch.from_numpy(x.transpose(2,0,1).copy()).float()[None]
  raw=seg(x)[0][0,0].numpy();prob=(raw-raw.min())/max(float(raw.max()-raw.min()),1e-12);raw320[g]=prob
  full=Image.fromarray(prob.astype(np.float32),mode='F').resize(im.size,Image.Resampling.BILINEAR)
  seed=int(hashlib.sha256(('20261120|'+g).encode()).hexdigest()[:8],16)
  for mode in modes:
   projected,valid=spatial(full,mode,mask=True);arr=np.clip(np.asarray(projected),0,1);patch=arr.reshape(16,14,16,14).mean((1,3)).reshape(-1)
   v=np.array(content[mode][h]['valid_patch_ids']);src=features[mode][h];selected=source_supports(patch,src[1:]@src[0],v,seed)
   assert len({len(z) for z in selected.values()})==1
   supports[mode][g]={'all':v,**selected};mask_vectors[mode][g]=patch
   mask_records.append({'mode':mode,'source_group':g,'valid_patches':len(v),'auto_K':len(selected['auto']),'auto_fraction_of_content_patches':len(selected['auto'])/len(v),'empty_fallback':bool((patch[v]>=.5).sum()==0)})
   if mode=='center':
    expected=np.array(m['selected_patches'],dtype=int)
    if not len(expected):expected=np.array([int(np.argmax(m['mask_patch_probabilities']))])
    np.testing.assert_array_equal(selected['auto'],expected)
  if index%10==0:print('source masks',index+1,'/46',flush=True)
 del seg
seg_seconds=time.perf_counter()-t_seg
np.savez_compressed(out/'source_masks_float320.npz',**raw320)
np.savez_compressed(out/'source_patch_probabilities.npz',**{mode+'_'+g:v for mode,groups in mask_vectors.items() for g,v in groups.items()})
(out/'supports_before_scores.json').write_text(json.dumps({mode:{g:{k:v.tolist() for k,v in regions.items()} for g,regions in groups.items()} for mode,groups in supports.items()},indent=2))
pd.DataFrame(mask_records).to_csv(out/'source_mask_metrics.csv',index=False)
records=[];missing=[]
for row in audit:
 scores={part:{} for part in ['A','B']};g=row['source_group']
 for mode in modes:
  src=features[mode][row['pixel_hashes']['source']]
  for part in ['A','B']:
   h=row['pixel_hashes'][part];dst=features[mode][h];valid=content[mode][h]['valid_patch_ids'];matrix=src[1:]@dst[1:].T
   scores[part][mode+'_global_CLS']={'score':float(src[0]@dst[0]),'valid':True,'unmatched_source_patches':0}
   for region,ids in supports[mode][g].items():
    for suffix,radius in [('free',None),('r2',2)]:
     result=nearest(matrix,ids,valid,radius);name=mode+'_'+region+'_'+suffix;scores[part][name]=result
     if not result['valid']:missing.append({'pair_id':row['pair_id'],'candidate':part,'method':name,**result})

 records.append({'pair_id':row['pair_id'],'row_idx':row['row_idx'],'source_group':g,'scores':scores})
(out/'scores_before_labels.json').write_text(json.dumps(records,indent=2,allow_nan=False));(out/'invalid_correspondences.json').write_text(json.dumps(missing,indent=2))
# New model outputs and source supports are sealed above before expert labels are read.
labels={r['pair_id']:r['expert_winner'][-1] for r in load(old/'protocol/inputs.json')['rows']}
prior_annotations={x['source_group']:x for x in load(root/'runs/20261120-mac-04/protocol/source_annotations_frozen.json')['rows']}
methods=list(records[0]['scores']['A']);rows=[];ordered=[]
for r in records:
 g=r['source_group'];an=prior_annotations[g];flat={k:r[k] for k in ['pair_id','row_idx','source_group']};flat.update({'expert_winner':labels[r['pair_id']],'prior_subject_eligible':an['subject_eligible'],'prior_crop_flag':an['crop_truncates_proposed_subject']})
 for name in methods:
  aa=r['scores']['A'][name];bb=r['scores']['B'][name]
  if not aa['valid'] or not bb['valid']:winner='invalid';credit=0.
  else:
   winner='A' if aa['score']>bb['score'] else 'B' if bb['score']>aa['score'] else 'tie';credit=.5 if winner=='tie' else float(winner==labels[r['pair_id']])
  flat[name+'_winner']=winner;flat[name+'_credit']=credit;flat[name+'_score_A']=aa['score'];flat[name+'_score_B']=bb['score']
  for order,ids in [('AB',['A','B']),('BA',['B','A'])]:
   x,y=[r['scores'][part][name] for part in ids]
   raw='Invalid' if winner=='invalid' else 'Image A' if x['score']>y['score'] else 'Image B' if y['score']>x['score'] else 'Tie'
   mapped='invalid' if raw=='Invalid' else ids[0] if raw=='Image A' else ids[1] if raw=='Image B' else 'tie';assert mapped==winner
   ordered.append({'pair_id':r['pair_id'],'source_group':g,'method':name,'order':order,'score_a':x['score'],'score_b':y['score'],'raw_output':raw,'mapped_winner':mapped,'label_pref':labels[r['pair_id']],'status':'invalid' if mapped=='invalid' else 'tie' if mapped=='tie' else 'ok'})
 rows.append(flat)
df=pd.DataFrame(rows);df.to_csv(out/'per_pair_results.csv',index=False);pd.DataFrame(ordered).to_csv(out/'ordered_logs.csv',index=False)
legacy=pd.read_csv(old/'outputs/per_pair_results.csv').set_index('pair_id')
mapping={'global_CLS':'B0_DINO_CLS','all_free':'B1_all_unrestricted','all_r2':'A_all_radius2','auto_free':'A_subject_unrestricted','auto_r2':'B3_subject_radius2'}
for key,previous in mapping.items():assert df.set_index('pair_id')['center_'+key+'_winner'].equals(legacy.loc[df.pair_id,previous+'_winner'])
for key in ['salient','random','center']:
 for suffix in ['free','r2']:
  previous='B2_'+key+'_'+('unrestricted' if suffix=='free' else 'radius2');assert df.set_index('pair_id')['center_'+key+'_'+suffix+'_winner'].equals(legacy.loc[df.pair_id,previous+'_winner'])
def paired(data,target,baseline):
 delta=data[target+'_credit']-data[baseline+'_credit'];blocks=data.assign(delta=delta).groupby('source_group').delta.agg(['sum','size']).to_numpy();rng=np.random.default_rng(20261120);chosen=blocks[rng.integers(len(blocks),size=(4000,len(blocks)))];bs=chosen[:,:,0].sum(axis=1)/chosen[:,:,1].sum(axis=1)
 return {'wins':int((delta>0).sum()),'new_errors':int((delta<0).sum()),'same_credit':int((delta==0).sum()),'delta':float(delta.mean()),'source_group_95ci':np.percentile(bs,[2.5,97.5]).tolist()}
comparisons=[]
for mode in modes:
 comparisons.extend([(mode+'_auto_r2',mode+'_all_free'),(mode+'_all_r2',mode+'_all_free'),(mode+'_auto_r2',mode+'_auto_free')])
for mode in modes[1:]:
 for name in mapping:comparisons.append((mode+'_'+name,'center_'+name))
subsets={}
for name,data in [('all89',df),('fixed_prior_eligible75',df[df.prior_subject_eligible]),('fixed_prior_crop_flag',df[df.prior_crop_flag]),('fixed_prior_no_crop_flag',df[~df.prior_crop_flag])]:
 subsets[name]={'pairs':len(data),'source_groups':data.source_group.nunique(),'methods':{m:{'correct':int((data[m+'_credit']==1).sum()),'ties':int((data[m+'_winner']=='tie').sum()),'invalid':int((data[m+'_winner']=='invalid').sum()),'accuracy':float(data[m+'_credit'].mean())} for m in methods},'paired':{target+'_vs_'+base:paired(data,target,base) for target,base in comparisons}}
if word is not None:assert sha(word)==input_seal['Word_sha256_before']
summary={'run_id':R.name,'stage':'reused old diagnostic, no confirmation','subsets':subsets,'source_masks':{mode:{'empty_fallback_groups':sum(x['empty_fallback'] for x in mask_records if x['mode']==mode),'median_K':float(np.median([x['auto_K'] for x in mask_records if x['mode']==mode])),'median_valid_patches':float(np.median([x['valid_patches'] for x in mask_records if x['mode']==mode]))} for mode in modes},'center_replay':{'feature_max_abs_error':maxerr,'auto_supports_match_all46':True,'all11_method_winners_match_all89':True},'invalid_correspondence_candidates':len(missing),'order_inconsistency':0,'timings':{'DINO_unique_forward_seconds':forward_time,'shared_source_segmentation_seconds':seg_seconds,'pipeline_wall_seconds_since_feature_encoding_start':time.perf_counter()-encode_begin,'note':'CPU2, batched model forward totals; not per-request latency or a PVC speed comparison'},'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'timm':timm.__version__,'device':'CPU','threads':2},'Word_unchanged':True if word is not None else None,'GPU_or_paid_resources':False,'confirmation_scored':False,'PVC_run':False,'limits':['All33 methods and3 geometries evaluated on previously inspected diagnostic pairs.','Letterbox changes content resolution and introduces padding; pure-padding matches excluded but attention still sees padding.','Square warp changes aspect ratio; this sensitivity does not isolate cropping causally.','Radius uses fixed canvas grid; content support differences may produce explicit invalid matches.','No new expert masks or independent confirmation, no novelty or PVC speed claim.']}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({'all89':{k:{name:subsets['all89']['methods'][k+'_'+name] for name in mapping} for k in modes},'center_replay':summary['center_replay'],'invalid_candidates':len(missing),'source_masks':summary['source_masks']},indent=2),flush=True)
