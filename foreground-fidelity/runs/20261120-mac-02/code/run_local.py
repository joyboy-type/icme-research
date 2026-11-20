import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import sys,json,time,hashlib,platform,copy
import numpy as np,pandas as pd,torch,timm
from PIL import Image
from skimage import transform as sktransform
from safetensors.torch import load_file
from timm.data import create_transform,resolve_model_data_config
from torchvision import transforms
from correspondence import mean_nearest,supports
from u2net_vendor import U2NETP
R=Path(__file__).resolve().parents[1];ROOT=R.parents[1];OUT=R/'outputs';DATA=ROOT/'data'
OUT.mkdir(exist_ok=True);(OUT/'source_masks').mkdir(exist_ok=True)
torch.set_num_threads(2);torch.set_num_interop_threads(2)
rows=json.loads((R/'protocol/inputs.json').read_text())['rows']
# Retain labels separately; they cannot affect mask, feature or support calculations.
labels={r['pair_id']:r.pop('expert_winner')[-1] for r in rows}
audit=[];groups={};canonical={};pixel_seen={}
for r in rows:
    r['pixel_hashes']={};r['file_hashes']={};r['dimensions']={}
    for part,filename in r['images'].items():
        q=Path(filename)
        with Image.open(q) as im:
            rgb=im.convert('RGB');wh=rgb.size
            digest=hashlib.sha256(str(wh).encode()+rgb.tobytes()).hexdigest()
            # Legacy grouping used RGB pixels alone; preserve exact group IDs for replay.
            legacy=hashlib.sha256(rgb.tobytes()).hexdigest()
        r['pixel_hashes'][part]=digest;r['file_hashes'][part]=hashlib.sha256(q.read_bytes()).hexdigest();r['dimensions'][part]=wh
        canonical.setdefault(digest,q)
        if part=='source':
            r['source_group']=legacy;groups.setdefault(legacy,digest)
    audit.append(copy.deepcopy(r))
assert len(rows)==89 and len(groups)==46
(OUT/'input_audit_before_scoring.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
model=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=False,dynamic_img_size=True)
dino_weight=DATA/'snapshot-01/weights/dinov2_vits14.safetensors'
assert hashlib.sha256(dino_weight.read_bytes()).hexdigest()=='04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081'
model.load_state_dict(load_file(str(dino_weight)));model.eval()
config=resolve_model_data_config(model);config['input_size']=(3,224,224)
pre=create_transform(**config,is_training=False)
spatial=[t for t in pre.transforms if isinstance(t,(transforms.Resize,transforms.CenterCrop))]
(OUT/'preprocessing.txt').write_text(str(pre)+'\nSource mask uses the same Resize/CenterCrop before14x14 patch averaging.\n')
features={};encode_times={};started=time.perf_counter()
with torch.inference_mode():
    items=list(canonical.items())
    for i in range(0,len(items),6):
        batch=items[i:i+6];x=torch.stack([pre(Image.open(q).convert('RGB')) for _,q in batch])
        t=time.perf_counter();f=model.forward_features(x).float();f=f/(f.norm(dim=-1,keepdim=True)+1e-12);elapsed=time.perf_counter()-t
        for (h,_),ft in zip(batch,f.numpy()):features[h]=ft;encode_times[h]=elapsed/len(batch)
    del model
print('DINO encoded',len(features),'unique pixel inputs in',round(time.perf_counter()-started,2),'seconds',flush=True)
np.savez_compressed(OUT/'dino_features.npz',keys=np.array(list(features)),features=np.array(list(features.values())))
# Only source pixels are passed to the saliency model.
seg=U2NETP(3,1);seg_weight=DATA/'snapshot-02/weights/u2netp.pth'
assert hashlib.sha256(seg_weight.read_bytes()).hexdigest()=='e7567cde013fb64813973ce6e1ecc25a80c05c3ca7adbc5a54f3c3d90991b854'
seg.load_state_dict(torch.load(seg_weight,map_location='cpu',weights_only=True));seg.eval()
mask_probs={};mask_times={};mask_meta=[]
with torch.inference_mode():
    for index,(group,h) in enumerate(groups.items()):
        image=Image.open(canonical[h]).convert('RGB');array=np.asarray(image)
        t=time.perf_counter();x=sktransform.resize(array,(320,320),mode='constant')
        x=x/max(float(x.max()),1e-12);x=(x-np.array([.485,.456,.406]))/np.array([.229,.224,.225]);x=torch.from_numpy(x.transpose(2,0,1).copy()).float()[None]
        raw=seg(x)[0][0,0].numpy();prob=(raw-raw.min())/max(float(raw.max()-raw.min()),1e-12)
        full=Image.fromarray(prob.astype(np.float32),mode='F').resize(image.size,Image.Resampling.BILINEAR)
        full_png=Image.fromarray(np.uint8(np.clip(np.asarray(full),0,1)*255));full_png.save(OUT/'source_masks'/f'{group}.png')
        cropped=full
        for operation in spatial:cropped=operation(cropped)
        arr=np.clip(np.asarray(cropped),0,1);assert arr.shape==(224,224)
        patch=arr.reshape(16,14,16,14).mean((1,3)).reshape(-1)
        mask_probs[group]=patch;mask_times[group]=time.perf_counter()-t
        mask_meta.append({'source_group':group,'source_pixel_sha':h,'source_file':str(canonical[h]),'mask_patch_probabilities':patch.tolist(),'selected_patches':np.flatnonzero(patch>=.5).tolist(),'area_fraction_at_dino_crop':float((patch>=.5).mean()),'empty_fallback':bool((patch>=.5).sum()==0),'selection_ms':1000*mask_times[group]})
        if index%10==0:print('source masks',index+1,'/',len(groups),flush=True)
    del seg
(OUT/'source_mask_metadata_before_labels.json').write_text(json.dumps(mask_meta,indent=2))
methods={'B0_DINO_CLS':None,'B1_all_unrestricted':None,'A_all_radius2':None,'A_subject_unrestricted':None,'B3_subject_radius2':None,
         **{f'B2_{key}_{suffix}':None for key in ['salient','random','center'] for suffix in ['unrestricted','radius2']},'legacy_salient_quarter':None}
score_records=[];latency_records=[];source_supports={}
for row in rows:
    group=row['source_group'];src=features[row['pixel_hashes']['source']];n=len(src)-1
    seed=int(hashlib.sha256(('20261120|'+group).encode()).hexdigest()[:8],16)
    support=supports(mask_probs[group],src,seed);assert len({len(x) for x in support.values()})==1
    source_supports[group]={k:v.tolist() for k,v in support.items()};scores={}
    for part in ['A','B']:
        dst=features[row['pixel_hashes'][part]];matrix=src[1:]@dst[1:].T;all_ids=np.arange(n)
        t=time.perf_counter();sc={'B0_DINO_CLS':float(src[0]@dst[0]),'B1_all_unrestricted':mean_nearest(matrix,all_ids),
                                  'A_all_radius2':mean_nearest(matrix,all_ids,2),'A_subject_unrestricted':mean_nearest(matrix,support['subject']),
                                  'B3_subject_radius2':mean_nearest(matrix,support['subject'],2)}
        for key in ['salient','random','center']:
            sc[f'B2_{key}_unrestricted']=mean_nearest(matrix,support[key]);sc[f'B2_{key}_radius2']=mean_nearest(matrix,support[key],2)
        q=np.argsort(-(src[1:]@src[0]),kind='stable')[:n//4];sc['legacy_salient_quarter']=mean_nearest(matrix,q)
        assert set(sc)==set(methods) and np.isfinite(list(sc.values())).all();scores[part]=sc
        latency_records.append({'pair_id':row['pair_id'],'candidate':part,'head_all_methods_ms':1000*(time.perf_counter()-t)})
    score_records.append({'pair_id':row['pair_id'],'row_idx':row['row_idx'],'source_group':group,'legacy_batch':row['legacy_batch'],
                          'subject_patch_count':len(support['subject']),'scores':scores,
                          'dino_triplet_amortized_batch_ms':1000*sum(encode_times[row['pixel_hashes'][k]] for k in ['source','A','B']),
                          'source_selection_ms':1000*mask_times[group]})
(OUT/'scores_before_labels.json').write_text(json.dumps(score_records,indent=2))
(OUT/'supports_before_labels.json').write_text(json.dumps(source_supports,indent=2))
# Scoring begins only after source support and all predictions are serialized.
def winner(a,b):return 'A' if a>b else 'B' if b>a else 'tie'
flat=[];logs=[]
for r in score_records:
    gt=labels[r['pair_id']];out={k:v for k,v in r.items() if k!='scores'};out['expert_winner']=gt
    for name in methods:
        aa=r['scores']['A'][name];bb=r['scores']['B'][name];prediction=winner(aa,bb)
        out[name+'_score_a']=aa;out[name+'_score_b']=bb;out[name+'_winner']=prediction;out[name+'_credit']=.5 if prediction=='tie' else float(prediction==gt)
        flipped_raw=winner(bb,aa);flipped_mapped={'A':'B','B':'A','tie':'tie'}[flipped_raw];assert flipped_mapped==prediction
        for order,raw,sa,sb in [('A|B',prediction,aa,bb),('B|A',flipped_raw,bb,aa)]:
            logs.append({'run_id':R.name,'pair_id':r['pair_id'],'source_group':r['source_group'],'split':'legacy_diagnostic',
                         'method':name,'candidate_a_id':r['pair_id']+':A','candidate_b_id':r['pair_id']+':B','order':order,
                         'score_a':sa,'score_b':sb,'raw_output':raw,'status':'tie' if prediction=='tie' else 'ok','label_pref':gt,
                         'mapped_winner':prediction,'peak_vram_mb':0,'source_selection_ms':r['source_selection_ms'],
                         'elapsed_ms':r['dino_triplet_amortized_batch_ms']+(r['source_selection_ms'] if name.startswith(('B3','A_subject','B2')) else 0),
                         'timing_kind':'amortized CPU batch estimate; includes source selection for region controls; not single-request latency'})
    flat.append(out)
df=pd.DataFrame(flat);df.to_csv(OUT/'per_pair_results.csv',index=False);pd.DataFrame(logs).to_csv(OUT/'ordered_logs.csv',index=False);pd.DataFrame(latency_records).to_csv(OUT/'head_timings.csv',index=False)
summary={'run_id':R.name,'stage':'legacy diagnostic; no fresh confirmation scored','pairs':len(df),'source_groups':df.source_group.nunique(),'methods':{},'B3_paired':{},'order_inconsistency':0,'invalid_outputs':0,'mask_empty_fallback_groups':sum(r['empty_fallback'] for r in mask_meta),'source_mask_area':{'min':min(r['area_fraction_at_dino_crop'] for r in mask_meta),'median':float(np.median([r['area_fraction_at_dino_crop'] for r in mask_meta])),'max':max(r['area_fraction_at_dino_crop'] for r in mask_meta)},'time':{'DINO_unique_input_forward_seconds':sum(encode_times.values()),'source_selection_total_seconds':sum(mask_times.values()),'B3_source_selection_ms_median':float(np.median(list(mask_times.values()))*1000),'B3_source_selection_ms_p95':float(np.percentile(list(mask_times.values()),95)*1000)},'environment':{'python':platform.python_version(),'torch':torch.__version__,'timm':timm.__version__,'threads':torch.get_num_threads(),'device':'cpu'}}
for name in methods:
    summary['methods'][name]={'correct':int((df[name+'_credit']==1).sum()),'ties':int((df[name+'_credit']==.5).sum()),'accuracy':float(df[name+'_credit'].mean())}
for name in ['B0_DINO_CLS','B1_all_unrestricted','A_all_radius2','A_subject_unrestricted','B2_salient_radius2','B2_random_radius2','B2_center_radius2']:
    delta=df.B3_subject_radius2_credit-df[name+'_credit'];blocks=df.assign(delta=delta).groupby('source_group').delta.agg(['sum','size']).to_numpy();rng=np.random.default_rng(20261120);bs=[]
    for _ in range(4000):
        selected=blocks[rng.integers(len(blocks),size=len(blocks))];bs.append(selected[:,0].sum()/selected[:,1].sum())
    summary['B3_paired'][name]={'delta':float(delta.mean()),'wins':int((delta>0).sum()),'losses':int((delta<0).sum()),'ties':int((delta==0).sum()),'source_group_95ci':np.percentile(bs,[2.5,97.5]).tolist()}
assert summary['methods']['B0_DINO_CLS']['correct']==68 and summary['methods']['B1_all_unrestricted']['correct']==77 and summary['methods']['legacy_salient_quarter']['correct']==70
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
