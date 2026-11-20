"""Run fixed nuisance diagnostics on historical development images only."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
from collections import Counter
from time import perf_counter
import argparse,csv,io,json,sys
import cv2,numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from registration import estimate_translation,translate,residual_scores,calibration_threshold


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[3]);args=parser.parse_args()
    run=Path(__file__).resolve().parents[1];out=run/'outputs';out.mkdir(exist_ok=True)
    project=args.project.resolve();protocol=json.loads((run/'protocol.json').read_text());manifest=run/'inputs/development_manifest.json'
    assert sha256(manifest.read_bytes()).hexdigest()==protocol['inputs']['manifest_sha256']
    rows=json.loads(manifest.read_text());assert len(rows)==240 and all(r['split']=='development' for r in rows)
    # Frozen Word files are read for integrity only, never opened for editing.
    for p,h in protocol.get('word_guard',{}).items():assert sha256(Path(p).read_bytes()).hexdigest()==h
    cv2.setNumThreads(1);started=perf_counter()
    labels=np.asarray([r['label'] for r in rows]);groups=np.asarray([r['source_group'] for r in rows]);folds=np.zeros(len(rows),int)
    assert len(set(groups))==240,'This fixed pilot expects unique original-image groups'
    splits=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261120).split(np.zeros(len(rows)),labels,groups))
    for fold,(tr,te) in enumerate(splits):
        assert set(groups[tr]).isdisjoint(groups[te]);folds[te]=fold
    dump(out/'fold_assignments.json',[{'pair_id':r['id'],'group':r['source_group'],'fold':int(f)} for r,f in zip(rows,folds)])
    hashes={'protocol':sha256((run/'protocol.json').read_bytes()).hexdigest(),'manifest':sha256(manifest.read_bytes()).hexdigest(),
            'code':{p.name:sha256(p.read_bytes()).hexdigest() for p in sorted((run/'code').glob('*.py'))}}
    dump(run/'development_design_lock.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'hashes':hashes,'scope':'development only; no confirmation examples read'})
    scores=[]
    for i,row in enumerate(rows):
        images=[]
        for part in ['before','after']:
            image=np.asarray(Image.open(project/row[part+'_path']).convert('RGB'))
            assert sha256(image.tobytes()).hexdigest()==row[part+'_sha256'];images.append(image)
        before,after=images
        def conditions(image):
            buffer=io.BytesIO();Image.fromarray(image).save(buffer,format='JPEG',quality=95)
            jpeg=np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert('RGB'))
            return {'identity':image,'clean':image,'jpeg95':jpeg,'shift_x1':translate(image,1,0),'shift_y1':translate(image,0,1),
                    'shift_xy1':translate(image,1,1),'jpeg95_shift_x1':translate(jpeg,1,0)}
        negative=conditions(before);positive=conditions(after)
        for is_positive,bank,kinds in [(False,negative,protocol['negative_conditions']),(True,positive,protocol['positive_conditions'])]:
            for condition in kinds:
                target=bank[condition]
                for mode in protocol['registration']:
                    t=perf_counter();alignment=estimate_translation(before,target,mode,protocol['registration_rules'])
                    aligned=translate(target,*[-v for v in alignment['shift']])
                    values=residual_scores(before,aligned,protocol['common_border_exclusion_pixels'])
                    ms=(perf_counter()-t)*1000
                    for kind,value in values.items():
                        scores.append({'run_id':run.name,'pair_id':row['id'],'source_group':row['source_group'],'fold':int(folds[i]),'positive':int(is_positive),
                             'condition':condition,'label':row['label'] if is_positive else 'no_diff','method':mode+'_'+kind,'score':value,
                             'estimated_dx':alignment['shift'][0],'estimated_dy':alignment['shift'][1],'alignment_reason':alignment['reason'],
                             'phase_response':alignment['response'],'valid_tiles':alignment['valid_tiles'],'supporting_tiles':alignment['supporting_tiles'],
                             'covered_quadrants':alignment['covered_quadrants'],'joint_two_residuals_ms':ms})
        if (i+1)%20==0:print('scored originals',i+1,'/',len(rows),'seconds',round(perf_counter()-started,1),flush=True)
    methods=sorted({r['method'] for r in scores});thresholds=[]
    for method in methods:
        for fold in range(3):
            calibration=[r for r in scores if r['method']==method and r['fold']!=fold and not r['positive']]
            by={c:[r['score'] for r in calibration if r['condition']==c] for c in protocol['negative_conditions']}
            threshold=calibration_threshold(by)
            thresholds.append({'method':method,'held_out_fold':fold,'threshold':threshold,'calibration_originals':len({r['pair_id'] for r in calibration}),
                               'calibration_fpr_by_condition':{c:float(np.mean(np.asarray(v)>threshold)) for c,v in by.items()}})
            for r in scores:
                if r['method']==method and r['fold']==fold:r.update(threshold=threshold,predicted_change=int(r['score']>threshold))
    dump(out/'fold_thresholds.json',thresholds)
    with (out/'per_case_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
    summary={}
    for method in methods:
        subset=[r for r in scores if r['method']==method]
        positive={c:float(np.mean([r['predicted_change'] for r in subset if r['positive'] and r['condition']==c])) for c in protocol['positive_conditions']}
        negative={c:float(np.mean([r['predicted_change'] for r in subset if not r['positive'] and r['condition']==c])) for c in protocol['negative_conditions']}
        classes={label:{c:float(np.mean([r['predicted_change'] for r in subset if r['positive'] and r['condition']==c and r['label']==label])) for c in protocol['positive_conditions']} for label in sorted(set(labels))}
        clean=[r for r in subset if r['positive'] and r['condition']=='clean']
        summary[method]={'positive_recall':positive,'negative_fpr':negative,'worst_negative_fpr':max(negative.values()),'per_class_recall':classes,
                         'clean_positive_nonzero_alignment_count':sum(any([r['estimated_dx'],r['estimated_dy']]) for r in clean),
                         'joint_alignment_and_two_scores_ms_mean':float(np.mean([r['joint_two_residuals_ms'] for r in subset]))}
    # Paired bootstrap treats each original and all its derived conditions as one unit.
    paired={};rng=np.random.default_rng(71);draws=rng.integers(240,size=(2000,240))
    base_ids=[r['id'] for r in rows]
    for method in methods:
        if method=='none_pixel4':continue
        paired[method]={}
        for condition in protocol['positive_conditions']:
            def vector(m):
                found={r['pair_id']:r['predicted_change'] for r in scores if r['method']==m and r['positive'] and r['condition']==condition}
                return np.asarray([found[x] for x in base_ids],float)
            delta=vector(method)-vector('none_pixel4');values=delta[draws].mean(axis=1)
            paired[method][condition]={'recall_delta':float(delta.mean()),'conditional_95_ci':np.quantile(values,[.025,.975]).tolist()}
    # Quantify complete erasure of local positive residual before thresholding.
    retention={}
    raw={(r['pair_id'],r['condition']):r['score'] for r in scores if r['method']=='none_pixel4' and r['positive']}
    for method in ['global_phase_pixel4','tile_consensus_pixel4']:
        clean=[r for r in scores if r['method']==method and r['positive'] and r['condition']=='clean']
        retention[method]={'nonzero_raw_but_zero_aligned':sum(raw[(r['pair_id'],'clean')]>0 and r['score']==0 for r in clean),
                           'position_total':sum(r['label']=='position' for r in clean),
                           'position_zeroed':sum(r['label']=='position' and raw[(r['pair_id'],'clean')]>0 and r['score']==0 for r in clean)}
    result={'scope':'240 original development pairs; 3-fold OOF, no independent confirmation and no website-isolation claim',
            'originals':240,'derived_cases':2400,'score_rows':len(scores),'folds':3,'nuisance_target':'per-condition calibration FPR <=5%, observed OOF FPR reported',
            'metrics':summary,'paired_recall_difference_vs_raw_pixel4':paired,'local_change_retention':retention,'wall_seconds':perf_counter()-started,
            'interval_limit':'Conditional descriptive bootstrap on fixed OOF thresholds; not adjusted for trying six variants; not threshold-training uncertainty.'}
    dump(out/'summary.json',result)
    for p,h in protocol.get('word_guard',{}).items():assert sha256(Path(p).read_bytes()).hexdigest()==h
    print(json.dumps({'complete':True,'rows':len(scores),'seconds':round(perf_counter()-started,1),'recall':{m:v['positive_recall'] for m,v in summary.items()}}),flush=True)


if __name__=='__main__':main()
