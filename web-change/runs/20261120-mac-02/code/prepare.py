"""Acquire bounded official data, group before images, and seal labels."""
from pathlib import Path
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io,json,requests,cv2,numpy as np
from PIL import Image
from core import connected_groups

RUN=Path(__file__).resolve().parents[1]
PROJECT=RUN.parents[1]
OLD=PROJECT/'data/snapshot-01'
NEW=PROJECT/'data/snapshot-02'
REV='c6dd79d5e1c0cbb4e7ca234c9f53c418a75e30ce'

def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,str) and x.startswith('https://datasets-server.huggingface.co/'):
        return x.split('?')[0]
    return x

def get_rows(offset,length):
    url=f'https://datasets-server.huggingface.co/rows?dataset=tencent/DiffSpot&config=default&split=test&offset={offset}&length={length}'
    last=None
    for _ in range(3):
        try:
            r=requests.get(url,timeout=30)
            r.raise_for_status();data=r.json()
            for item in data['rows']:
                for field in ['image_before','image_after']:
                    assert REV in item['row'][field]['src'],'Image URL revision mismatch'
            return data
        except Exception as e:last=e
    raise last

def image_path(row,suffix):
    name=row['id']+'_'+suffix+'.img'
    a=OLD/'samples'/name
    return a if a.exists() else NEW/'samples'/name

def download(item):
    try:
        r=item['row']
        for suffix in ['before','after']:
            path=image_path(r,suffix)
            if not path.exists():
                response=requests.get(r['image_'+suffix]['src'],timeout=25);response.raise_for_status()
                im=Image.open(io.BytesIO(response.content));im.verify()
                path.write_bytes(response.content)
        return item,None
    except Exception as e:return item,str(e).split('?')[0]

def signatures(row):
    im=np.asarray(Image.open(image_path(row,'before')).convert('RGB'))
    small=cv2.resize(im,(64,40),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    gray=cv2.resize(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    dct=cv2.dct(gray)[:8,:8].ravel()[1:]
    phash=dct>np.median(dct)
    dh=cv2.resize(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),(17,16),interpolation=cv2.INTER_AREA)
    dhash=(dh[:,1:]>dh[:,:-1]).ravel()
    return sha256(im.tobytes()).hexdigest(),small,phash,dhash

def main():
    NEW.mkdir(parents=True,exist_ok=True);(NEW/'samples').mkdir(exist_ok=True)
    def check_revision():
        response=requests.get('https://huggingface.co/api/datasets/tencent/DiffSpot',timeout=25)
        response.raise_for_status();revision=response.json()['sha']
        assert revision==REV,('Dataset revision changed',revision,REV)
        return revision
    revision_before=check_revision()
    historical=[]
    for part,f in [('train','first_rows.json'),('train','confirm_rows.json'),('quarantine','edge_holdout_rows.json')]:
        for item in json.loads((OLD/f).read_text())['rows']:
            item['origin']=part;historical.append(item)
    old_ids={x['row']['id'] for x in historical}
    candidates=[]
    # Offsets are frozen before acquisition and do not depend on candidate labels.
    for offset in [100,1000,1300,2200,2500,3400,3700,3900]:
        data=get_rows(offset,60 if offset!=3900 else 80)
        saved=NEW/('rows_%04d.json'%offset)
        if saved.exists():assert json.loads(saved.read_text())==clean(data),'Cached metadata changed'
        else:dump(saved,clean(data))
        for item in data['rows']:
            if item['row']['id'] not in old_ids:
                item['origin']='official_control' if offset==3900 else 'candidate'
                candidates.append(item)
        print('metadata',offset,len(data['rows']),flush=True)
    accepted=[];failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,(item,error) in enumerate(pool.map(download,candidates),1):
            if error:failures.append({'id':item['row']['id'],'reason':error})
            else:accepted.append(item)
            if n%50==0:print('downloaded',n,'failed',len(failures),flush=True)
    # Historical unavailable pair is excluded without retrying its signed URL.
    all_items=historical+accepted
    usable=[]
    for item in all_items:
        r=item['row']
        # Incomplete historical pairs still exclude their available before image.
        if image_path(r,'before').exists():usable.append(item)
    # One node per original image, retaining all mutation pairs for grouping.
    sigs=[signatures(x['row']) for x in usable]
    edges=[];near=[]
    for i,(h,small,ph,dh) in enumerate(sigs):
        for j in range(i):
            h2,s2,p2,d2=sigs[j]
            hp=int(np.count_nonzero(ph!=p2));hd=int(np.count_nonzero(dh!=d2))
            if h==h2:
                edges.append((i,j));continue
            if hp<=8 and hd<=24:
                mae=float(np.abs(small-s2).mean())
                if mae<=.025:
                    edges.append((i,j));near.append({'id_a':usable[i]['row']['id'],'id_b':usable[j]['row']['id'],'phash_distance':hp,'dhash_distance':hd,'rgb_mae':mae})
    groups=connected_groups([x['row']['id'] for x in usable],edges)
    old_groups={g for x,g in zip(usable,groups) if x['origin'] in ['train','quarantine']}
    seen=set();manifest=[];exclusions=[];labels={}
    for item,g,sig in zip(usable,groups,sigs):
        r=item['row'];origin=item['origin'];h=sig[0]
        if not image_path(r,'after').exists():
            exclusions.append({'id':r['id'],'reason':'after_missing_before_retained_as_exclusion_anchor','group':g});continue
        if origin in ['candidate','official_control'] and g in old_groups:
            exclusions.append({'id':r['id'],'reason':'near_duplicate_group_overlaps_history','group':g});continue
        elif h in seen:
            exclusions.append({'id':r['id'],'reason':'before_pixels_repeated','group':g});continue
        else:
            split='development' if origin=='train' else ('quarantine' if origin=='quarantine' else ('confirmation_control' if origin=='official_control' else 'confirmation'))
        seen.add(h)
        paths=[image_path(r,s) for s in ['before','after']]
        hashes=[]
        for p in paths:
            a=np.asarray(Image.open(p).convert('RGB'));hashes.append(sha256(a.tobytes()).hexdigest())
        manifest.append({'id':r['id'],'row_idx':item['row_idx'],'split':split,'source_group':g,'source_group_kind':'near_duplicate_proxy_not_website',
                         'source_url':'','domain_category':r['domain'],'before_sha256':hashes[0],'after_sha256':hashes[1],
                         'before_path':str(paths[0].relative_to(PROJECT)),'after_path':str(paths[1].relative_to(PROJECT))})
        labels[r['id']]={'label':r['mutation_types'][0].removeprefix('mutate_') if r['mutation_types'] else 'no_diff','difficulty':r['difficulty'],'task_type':r['task_type']}
    dump(NEW/'manifest.json',manifest)
    dump(NEW/'labels.sealed.json',labels)
    dump(NEW/'exclusions.json',exclusions+failures)
    dump(RUN/'outputs/grouping_audit.json',{'official_source_identifiers':False,'source_url_available':False,'rule':'connected components; exact before RGB hash OR pHash distance <=8 AND dHash distance <=24 AND RGB64x40 MAE <=.025',
         'warning':'No claim of website isolation. Thresholds fixed before candidate labels were read. Similarity can miss same-source redesigns or merge unrelated templates.',
         'dataset_revision_before':revision_before,'dataset_revision_after':check_revision(),
         'historical_before_anchors':len(historical),'fresh_available_pairs':len(accepted),'splits':{s:sum(x['split']==s for x in manifest) for s in sorted({x['split'] for x in manifest})},
         'near_duplicate_edges':len(near),'near_duplicate_examples':near[:30],'groups':len(set(groups)),'download_failures':failures})
    dump(RUN/'outputs/grouping_edges.json',near)
    print(json.dumps(json.loads((RUN/'outputs/grouping_audit.json').read_text()),ensure_ascii=False)[:1400],flush=True)

if __name__=='__main__':main()
