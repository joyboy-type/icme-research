from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json,requests
from source_utils import sanitize,git_blob_sha

RUN=Path(__file__).resolve().parents[1];PROJECT=RUN.parents[1];DATA=PROJECT/'data/template-probes-01'
def dump(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    DATA.mkdir(parents=True,exist_ok=False);(DATA/'sources').mkdir();(DATA/'pages').mkdir();(DATA/'images').mkdir()
    specs=json.loads((RUN/'source_specs.json').read_text())['families'];protocol=json.loads((RUN/'protocol.json').read_text())
    def one(s):
        folder=DATA/'sources'/s['family'];folder.mkdir();files=[]
        def get(path,name,url=None,expected_sha256=None):
            address=url or f"https://raw.githubusercontent.com/{s['repository']}/{s['commit_sha']}/{path}"
            last=None
            for _ in range(3):
                try:
                    response=requests.get(address,timeout=25);response.raise_for_status();raw=response.content
                    if path:assert git_blob_sha(raw)==s['source_blobs'][path],('Upstream blob changed',s['family'],path)
                    elif expected_sha256:assert sha256(raw).hexdigest()==expected_sha256,('CDN bytes changed',s['family'],name)
                    (folder/name).write_bytes(raw)
                    files.append({'path':str((folder/name).relative_to(DATA)),'url':address,'sha256':sha256(raw).hexdigest(),'upstream_blob_sha':s['source_blobs'].get(path)})
                    return raw.decode('utf-8-sig')
                except Exception as e:last=e
            raise last
        license=get(s['license'],'LICENSE.txt')
        if not ('Permission is hereby granted' in license or 'Redistribution and use' in license):raise ValueError('Unrecognized license '+s['family'])
        html=get(s['html'],'original.html');styles=[]
        for item in s['css']:styles.append(get(item.get('path'),item['name'],item.get('url'),item.get('expected_sha256')))
        clean,audit=sanitize(html,'\n'.join(styles));page=DATA/'pages'/(s['family']+'.html');page.write_text(clean)
        return {**s,'split':protocol['family_split'][s['family']],'files':files,'page':str(page.relative_to(DATA)),'page_sha256':sha256(page.read_bytes()).hexdigest(),'audit':audit}
    results=[];errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[(s,pool.submit(one,s)) for s in specs]
        for spec,f in futures:
            try:r=f.result();results.append(r);print('source ready',spec['family'],flush=True)
            except Exception as e:errors.append({'family':spec['family'],'error':str(e)});print('source failed',spec['family'],str(e),flush=True)
    dump(DATA/'sources.json',results);dump(RUN/'outputs/acquisition_errors.json',errors)
    assert len(results)==12,errors
    git_files=sum(bool(f['upstream_blob_sha']) for s in results for f in s['files']);cdn_files=sum(not f['upstream_blob_sha'] for s in results for f in s['files'])
    dump(RUN/'outputs/source_acquisition.json',{'families':12,'git_blob_verified_files':git_files,'versioned_cdn_sha256_verified_files':cdn_files,'data':'data/template-probes-01','source_manifest_sha256':sha256((DATA/'sources.json').read_bytes()).hexdigest()})

if __name__=='__main__':main()
