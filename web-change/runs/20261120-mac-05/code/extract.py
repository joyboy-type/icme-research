"""Fixed image-only features. No label, DOM target or annotation is used in descriptors."""
from pathlib import Path
from hashlib import sha256
from time import perf_counter
import argparse,io,json,os
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
import cv2,numpy as np
from PIL import Image
from core import crop_xyxy,edge_blocks
from legacy_paired import summarize,projection_stats
from registration import estimate_translation,translate
from local_features import describe
from evaluation_utils import unique_baselines,file_hashes
RUN=Path(__file__).resolve().parents[1];PROJECT=RUN.parents[1];DATA=PROJECT/'data/template-probes-01';OUT=RUN/'outputs'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def image(p):return np.asarray(Image.open(PROJECT/p).convert('RGB'))
def pixel_features(a,b,rect):
 h,w=a.shape[:2];absolute=np.abs(a.astype(np.float32)-b.astype(np.float32));delta=absolute.max(axis=2)
 ys,xs=np.where(delta>4)
 if not len(xs):ys,xs=np.where(delta>0)
 y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
 x,y,x1c,y1c=rect;crop=(slice(y,y1c),slice(x,x1c));d=delta[crop];m=d>4
 if not m.any():m=d>0
 base=[(y1-y0)/h,(x1-x0)/w,np.sqrt(len(xs)/(h*w)),len(xs)/((y1-y0)*(x1-x0))]
 base+=summarize(d/m.size)+summarize(d[m]/255)+projection_stats(m.astype(np.float32))
 for channel in range(3):base+=summarize(absolute[crop][:,:,channel][m]/255)
 return np.r_[base,cv2.resize(d,(16,16),interpolation=cv2.INTER_AREA).ravel()/255]
def variant(a,c):
 if c in ['identity','clean']:return a
 if c=='shift_x1':return translate(a,1,0)
 if c.startswith('jpeg'):
  quality=int(c[4:].split('_')[0]);f=io.BytesIO();Image.fromarray(a).save(f,format='JPEG',quality=quality)
  b=np.asarray(Image.open(io.BytesIO(f.getvalue())).convert('RGB'))
  return translate(b,1,0) if 'shift_x1' in c else b
 raise ValueError(c)
def detector():
 started=perf_counter();rows=read(DATA/'manifest.json');baselines=unique_baselines(rows);declared=read(DATA/'baselines.json')
 # Includes baselines with no valid modifications; controls remain independent of positive eligibility.
 known={r['id'] for r in baselines}
 baselines.extend({k:r[k] for k in ['id','family','split','before_path','before_sha256']} for r in declared if r['id'] not in known)
 assert len({r['before_sha256'] for r in baselines})==len(baselines)
 cfg=read(RUN/'evaluation_config.json');features=[];scalars=[];records=[]
 def add(a,b,meta):
  alignment=estimate_translation(a,b,'global_phase');aligned=translate(b,*[-v for v in alignment['shift']]);v,n,s=describe(a,aligned)
  features.append(v);scalars.append([s['area_blur8'],s['max_local32']]);records.append({**meta,'estimated_shift':alignment['shift'],'alignment_reason':alignment['reason']});return n
 for r in baselines:
  a=image(r['before_path']);assert sha256(a.tobytes()).hexdigest()==r['before_sha256']
  for c in cfg['negative_conditions']:names=add(a,variant(a,c),{'id':r['id']+'__negative__'+c,'baseline_id':r['id'],'family':r['family'],'split':r['split'],'label':'no_diff','positive':0,'condition':c})
 for i,r in enumerate(rows):
  a,b=image(r['before_path']),image(r['after_path']);assert sha256(b.tobytes()).hexdigest()==r['after_sha256']
  for c in cfg['positive_detection_conditions']:names=add(a,variant(b,c),{'id':r['id']+'__positive__'+c,'pair_id':r['id'],'baseline_id':r['baseline_id'],'family':r['family'],'split':r['split'],'label':r['label'],'positive':1,'condition':c})
  if (i+1)%50==0:print('detector features',i+1,'/',len(rows),'seconds',round(perf_counter()-started,1),flush=True)
 x=np.asarray(features);assert np.isfinite(x).all();np.savez_compressed(OUT/'detection_features.npz',x=x,scalar=np.asarray(scalars))
 dump(OUT/'detection_index.json',records);dump(OUT/'feature_names.json',names);dump(OUT/'detection_extraction.json',{'seconds':perf_counter()-started,'rows':len(records),'positive_pairs':len(rows),'unique_control_baselines':len(baselines),'features':x.shape[1],'no_annotation_features':True})
 print('detector extraction complete',len(records),flush=True)
def classification(weights):
 import torch
 from transformers import Dinov2Model
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 started=perf_counter();rows=read(DATA/'manifest.json');hog=cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
 values={k:[] for k in ['pixel_ET','pixel_HOG_ET','pixel_edges_ET']};images=[];crops=[]
 for i,r in enumerate(rows):
  a,b=image(r['before_path']),image(r['after_path']);rect=crop_xyxy(a,b);assert rect is not None
  x,y,x1,y1=rect;ca,cb=a[y:y1,x:x1],b[y:y1,x:x1];base=pixel_features(a,b,rect)
  ha,hb=[hog.compute(cv2.resize(cv2.cvtColor(c,cv2.COLOR_RGB2GRAY),(64,64))).ravel() for c in [ca,cb]]
  ea,eb=[cv2.Canny(cv2.cvtColor(c,cv2.COLOR_RGB2GRAY),40,100)>0 for c in [ca,cb]]
  values['pixel_ET'].append(base);values['pixel_HOG_ET'].append(np.r_[base,np.abs(hb-ha)]);values['pixel_edges_ET'].append(np.r_[base,edge_blocks(ea,eb,'directional')])
  images.append([cv2.resize(c,(224,224),interpolation=cv2.INTER_AREA) for c in [ca,cb]]);crops.append(rect)
 model=Dinov2Model.from_pretrained(weights,local_files_only=True).eval();load_seconds=perf_counter()-started
 mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None];contrasts=[]
 with torch.inference_mode():
  model(pixel_values=torch.zeros((2,3,224,224)))
  for pos in range(0,len(images),8):
   pixels=np.asarray(images[pos:pos+8]).reshape(-1,224,224,3);tensor=torch.from_numpy(pixels).permute(0,3,1,2).float()/255
   hidden=torch.nn.functional.normalize(model(pixel_values=(tensor-mean)/std).last_hidden_state,dim=-1).numpy()
   for before,after in zip(hidden[::2],hidden[1::2]):
    delta=after-before;contrasts.append(np.r_[before[0],after[0],delta[0],np.abs(delta[1:]).mean(axis=0),np.abs(delta[1:]).max(axis=0)])
   if pos%64==0:print('DINO encoded',len(contrasts),'/',len(images),'seconds',round(perf_counter()-started,1),flush=True)
 arrays={k:np.asarray(v) for k,v in values.items()};arrays['DINO_pixel_ET']=np.c_[arrays['pixel_ET'],np.asarray(contrasts)]
 assert all(np.isfinite(v).all() for v in arrays.values());np.savez_compressed(OUT/'classification_features.npz',**arrays)
 dump(OUT/'classification_index.json',[{**{k:r[k] for k in ['id','baseline_id','family','split','label','source_url','before_sha256','after_sha256']},'crop_xyxy':c} for r,c in zip(rows,crops)])
 dump(OUT/'classification_extraction.json',{'seconds':perf_counter()-started,'pairs':len(rows),'through_model_load_seconds':load_seconds,'device':'CPU','threads':2,'batch_pairs':8,'weights_sha256':file_hashes(list(weights.glob('*')),weights),'dimensions':{k:v.shape[1] for k,v in arrays.items()},'input':'RGB screenshots, shared pixel-difference crop; no DOM/bbox/label supplied to descriptor'})
 print('classification extraction complete',len(rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['detector','classification']);p.add_argument('--weights',type=Path,default=PROJECT.parents[1]/'work/paired_screen_pilot/dino');args=p.parse_args();cv2.setNumThreads(1)
 if args.stage=='detector':detector()
 else:classification(args.weights)
