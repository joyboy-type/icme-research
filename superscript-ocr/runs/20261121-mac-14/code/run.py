from pathlib import Path
import argparse,json,hashlib,time,datetime,platform
from merge import merge

def main():
    parser=argparse.ArgumentParser(description='Replay a cached two-engine OCR run; never calls OCR or modifies source data.')
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]/'20261121-mac-13')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/replay.json')
    args=parser.parse_args();p=Path(__file__).resolve().parents[1];assert not args.output.exists()
    sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
    protocol_hash=sha(p/'protocol.json');manifest=json.loads((p/'inputs/source_manifest.json').read_text())
    for entry in manifest['files']:assert sha(args.source/entry['relative_to_source'])==entry['sha256']
    pages=json.loads((args.source/'pages.json').read_text());raw=[json.loads(s) for s in (args.source/'outputs/whole_raw.jsonl').read_text().splitlines()];records=[];started=time.monotonic()
    for page in pages:
        byengine={r['engine_id']:r['observations'] for r in raw if r['case_id']==page['case_id']};assert len(byengine)==2
        for receiver_engine,donor_engine,base_label in [('vision_accurate_r3','tesseract','vision'),('tesseract','vision_accurate_r3','tesseract')]:
            receiver=byengine[receiver_engine];donor=byengine[donor_engine]
            records.append(dict(case_id=page['case_id'],method='single_'+base_label,receiver_engine=receiver_engine,donor_engine=None,lines=[dict(r,origin='receiver',receiver_index=i) for i,r in enumerate(receiver)],insertions=[],decisions=[],algorithm_seconds=0))
            for mode,position_check in [('text_only_merge',False),('position_neighbor_prototype',True)]:
                t=time.monotonic();result=merge(receiver,donor,page['width'],page['height'],position_check);elapsed=time.monotonic()-t
                originals=[r for r in result['lines'] if r['origin']=='receiver']
                assert len(originals)==len(receiver)
                for i,(before,after) in enumerate(zip(receiver,originals)):
                    assert {k:v for k,v in after.items() if k not in ['origin','receiver_index']}==before and after['receiver_index']==i
                records.append(dict(case_id=page['case_id'],method=mode+'_'+base_label+'_base',receiver_engine=receiver_engine,donor_engine=donor_engine,algorithm_seconds=elapsed,**result))
    payload=dict(id=p.name,data_role='run13_known_development_replay',source_run=args.source.name,protocol_sha256=protocol_hash,code_sha256={q.name:sha(q) for q in p.joinpath('code').glob('*.py')},records=records,resources=dict(cached_source_ocr_requests=8,new_ocr_calls=0,new_detector_calls=0,new_model_calls=0,new_automatic_pixel_verification_calls=0,new_automatic_pixel_verification_pixels=0,new_image_downloads=0,paid_calls=0,gpu_rental=False,training=False,algorithm_wall_seconds=time.monotonic()-started),system=platform.platform(),completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    assert sha(p/'protocol.json')==protocol_hash
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([dict(case_id=r['case_id'],method=r['method'],novel_candidates=sum(d['novel_candidate'] for d in r['decisions']),accepted=len(r['insertions']),seconds=round(r['algorithm_seconds'],4)) for r in records],indent=2))

if __name__=='__main__':main()
