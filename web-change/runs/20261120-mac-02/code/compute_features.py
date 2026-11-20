"""Extract shared-crop handcrafted and frozen DINO features without labels."""
from pathlib import Path
import os
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
from time import perf_counter
import io,json,cv2,numpy as np,torch
from PIL import Image
from transformers import Dinov2Model
from core import crop_xyxy,edge_blocks
from legacy_paired import summarize,projection_stats,features as legacy_features
from prepare import RUN,PROJECT,NEW,dump

MODES=['directional','merged','symmetric','new_only','old_only']
WEIGHTS=PROJECT.parents[1]/'work/paired_screen_pilot/dino'

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

def score(a,b):
    return float((np.abs(a.astype(np.int16)-b.astype(np.int16)).max(axis=2)>4).mean())

def main():
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    manifest=json.loads((NEW/'manifest.json').read_text())
    selected=[r for r in manifest if r['split'] in ['development','confirmation']]
    methods={'W0':[],'W1':[],**{k:[] for k in MODES}}
    crops=[];images=[];timings=[];negative=[];positive_scores=[]
    hog=cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
    for i,row in enumerate(selected):
        a,b=[np.asarray(Image.open(PROJECT/row[s+'_path']).convert('RGB')) for s in ['before','after']]
        t=perf_counter();rect=crop_xyxy(a,b);crop_ms=(perf_counter()-t)*1000
        if rect is None:raise ValueError('Unexpected unchanged positive '+row['id'])
        x,y,x1,y1=rect;ca,cb=a[y:y1,x:x1],b[y:y1,x:x1]
        t=perf_counter();base=pixel_features(a,b,rect);pixel_ms=(perf_counter()-t)*1000
        if i<3:
            original,full=legacy_features(a,b)
            np.testing.assert_allclose(base,original,rtol=1e-7,atol=1e-8)
        methods['W0'].append(base)
        t=perf_counter()
        ha,hb=[hog.compute(cv2.resize(cv2.cvtColor(c,cv2.COLOR_RGB2GRAY),(64,64))).ravel() for c in [ca,cb]]
        methods['W1'].append(np.r_[base,np.abs(hb-ha)])
        hog_ms=(perf_counter()-t)*1000
        t=perf_counter();ea,eb=[cv2.Canny(cv2.cvtColor(c,cv2.COLOR_RGB2GRAY),40,100)>0 for c in [ca,cb]]
        canny_ms=(perf_counter()-t)*1000
        desc_times={}
        for mode in MODES:
            t=perf_counter();block=edge_blocks(ea,eb,mode);desc_times[mode]=(perf_counter()-t)*1000
            methods[mode].append(np.r_[base,block])
            if mode=='directional' and i<3:np.testing.assert_allclose(block,full[len(base)+27:-6],rtol=1e-7,atol=1e-8)
        images.append([cv2.resize(c,(224,224),interpolation=cv2.INTER_AREA) for c in [ca,cb]])
        crops.append(rect)
        timings.append({'id':row['id'],'crop_ms':crop_ms,'pixel_ms':pixel_ms,'hog_increment_ms':hog_ms,'canny_ms':canny_ms,'descriptor_ms':desc_times})
        positive_scores.append(score(a,b))
        if row['split'] in ['development','confirmation']:
            buf=io.BytesIO();Image.fromarray(a).save(buf,format='JPEG',quality=95)
            jpeg=np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))
            shift=np.concatenate([a[:,:1],a[:,:-1]],axis=1)
            for kind,other in [('identity',a),('jpeg95',jpeg),('shift1',shift)]:
                negative.append({'id':row['id'],'split':row['split'],'source_group':row['source_group'],'condition':kind,'score':score(a,other)})
        if (i+1)%50==0:print('pixel features',i+1,'/',len(selected),flush=True)
    for row in manifest:
        if row['split']=='confirmation_control':
            a,b=[np.asarray(Image.open(PROJECT/row[s+'_path']).convert('RGB')) for s in ['before','after']]
            negative.append({'id':row['id'],'split':'confirmation','source_group':row['source_group'],'condition':'official_no_diff','score':score(a,b)})
    t=perf_counter();model=Dinov2Model.from_pretrained(WEIGHTS,local_files_only=True).eval()
    load_ms=(perf_counter()-t)*1000
    mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
    # Warmup is outside measured steady-state encoding; no training or adaptation.
    with torch.inference_mode():model(pixel_values=torch.zeros((2,3,224,224)))
    contrast=[];batch_times=[]
    with torch.inference_mode():
        for pos in range(0,len(images),8):
            t=perf_counter();pixels=np.asarray(images[pos:pos+8]).reshape(-1,224,224,3)
            tensor=torch.from_numpy(pixels).permute(0,3,1,2).float()/255
            hidden=model(pixel_values=(tensor-mean)/std).last_hidden_state
            hidden=torch.nn.functional.normalize(hidden,dim=-1).numpy()
            for before,after in zip(hidden[::2],hidden[1::2]):
                delta=after-before
                contrast.append(np.r_[before[0],after[0],delta[0],np.abs(delta[1:]).mean(axis=0),np.abs(delta[1:]).max(axis=0)])
            batch_times.extend([(perf_counter()-t)*1000/len(images[pos:pos+8])]*len(images[pos:pos+8]))
            if pos%64==0:print('DINO encoded',len(contrast),'/',len(images),flush=True)
    methods['dino']=contrast
    arrays={k:np.asarray(v) for k,v in methods.items()}
    arrays['dino_pixel']=np.c_[arrays['W0'],arrays['dino']]
    arrays['positive_score']=np.asarray(positive_scores)
    np.savez_compressed(RUN/'outputs/features.npz',**arrays)
    dump(RUN/'outputs/feature_index.json',[{**r,'crop_xyxy':c} for r,c in zip(selected,crops)])
    dump(RUN/'outputs/negative_scores.json',negative)
    dump(RUN/'outputs/timing_features.json',{'per_pair':timings,'dino_per_pair_amortized_ms':batch_times,'dino_model_load_ms':load_ms,'dino_size':224,'threads':2,'device':'CPU','warmup_pairs':1,'note':'Image IO excluded; crop shared. DINO batch8-pair amortized inference, not single-request latency.'})
    print('feature extraction complete',len(selected),flush=True)

if __name__=='__main__':main()
