from pathlib import Path
import json,subprocess,os,time,hashlib,platform

P=Path(__file__).resolve().parents[1];O=P/'outputs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def main():
    output=O/'whole_raw.jsonl';assert not output.exists()
    pages=json.loads((P/'pages.json').read_text());assert len(pages)==4
    initial={p['case_id']:sha(Path(p['path'])) for p in pages};protocol=sha(P/'protocol.json');started=time.monotonic()
    with output.open('w') as stdout,(O/'native.log').open('w') as stderr:
        proc=subprocess.run([str(P/'code/recognize'),*(p['path'] for p in pages)],stdout=stdout,stderr=stderr,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2'),check=False)
    results=[json.loads(s) for s in output.read_text().splitlines()];assert len(results)==12
    for page in pages:
        assert sha(Path(page['path']))==initial[page['case_id']]
        rows=[r for r in results if r['path']==page['path']];assert {r['configuration'] for r in rows}=={'accurate_r3','fast_r3','accurate_r2'}
        for r in rows:
            if r['status']=='completed':assert (r['pixelWidth'],r['pixelHeight'])==(page['width'],page['height'])
    assert sha(P/'protocol.json')==protocol
    meta=dict(id=P.name,status='whole_ocr_complete',pages=4,ocr_requests=12,successful_requests=sum(r['status']=='completed' for r in results),failed_requests=sum(r['status']=='error' for r in results),process_exit_code=proc.returncode,input_pixels=sum(p['width']*p['height'] for p in pages)*3,wall_seconds=time.monotonic()-started,actual_engine_count=1,configuration_count=3,independent_engines=False,system=platform.platform(),protocol_sha256=protocol,source_image_sha256=initial,code_sha256={p.name:sha(p) for p in (P/'code').glob('*') if p.suffix in ['.py','.swift']},model_downloads=0,paid_calls=0,training=False,gpu_rental=False,word_edit_allowed=False,automatic_insertion=False)
    save(P/'run.json',meta);print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
