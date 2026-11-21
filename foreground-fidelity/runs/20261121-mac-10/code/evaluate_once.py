"""Reproduce two cached heads for ten frozen cases; apply one rule before label scoring."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from trigger_core import route
p=argparse.ArgumentParser();p.add_argument('--experiment-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();root=a.experiment_root.resolve();run=a.run_root.resolve();prior=root/'runs/20261120-mac-02';started=time.perf_counter()
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
rule=json.loads((run/'protocol/trigger_rule_frozen.json').read_text())
for f,h in rule['frozen_input_hashes'].items():assert sha(run/f)==h
seal_path=run/'protocol/evaluation_seal_before_labels.json'
assert not seal_path.exists(), 'One evaluation only; do not overwrite an existing scored run'
seal={'rule_sha256':sha(run/'protocol/trigger_rule_frozen.json'),'eligibility_sha256':sha(run/'protocol/source_eligibility_frozen.json'),'selection_sha256':sha(run/'protocol/selection_PRIVATE.json'),'core_sha256':sha(Path(__file__).with_name('trigger_core.py')),'evaluator_sha256':sha(Path(__file__)),'label_columns_read_yet':False,'evaluation_number':1}
seal_path.write_text(json.dumps(seal,indent=2)+'\n')
cases=json.loads((run/'protocol/selection_PRIVATE.json').read_text())['selected'];assert len(cases)==10
eligibility={x['case_id']:x for x in json.loads((run/'protocol/source_eligibility_frozen.json').read_text())['records']}
supports=json.loads((prior/'outputs/supports_before_labels.json').read_text());cached={r['pair_id']:r for r in json.loads((prior/'outputs/scores_before_labels.json').read_text())}
archive=np.load(prior/'outputs/dino_features.npz');features=dict(zip(archive['keys'].tolist(),archive['features']));archive.close();records=[];maxerror=0
winner=lambda x,y:'A' if x>y else 'B' if y>x else 'tie'
for r in cases:
 metadata=r['metadata'];src=features[metadata['pixel_hashes']['source']][1:];n=len(src);ids=np.array(supports[r['source_group']]['subject'],dtype=int);other=np.setdiff1d(np.arange(n),ids);assert len(ids) and len(other)
 scores={}
 for part in ['A','B']:
  dst=features[metadata['pixel_hashes'][part]][1:];matches=(src@dst.T).max(axis=1)
  scores[part]={'all':float(matches.mean()),'subject':float(matches[ids].mean()),'complement':float(matches[other].mean())}
  for field,name in [('all','B1_all_unrestricted'),('subject','A_subject_unrestricted')]:
   error=abs(scores[part][field]-cached[r['pair_id']]['scores'][part][name]);maxerror=max(maxerror,error);assert error<=rule['score_reproduction_tolerance']
 b=winner(scores['A']['all'],scores['B']['all']);s=winner(scores['A']['subject'],scores['B']['subject']);assert b==r['B1_all_unrestricted_winner'] and s==r['A_subject_unrestricted_winner']
 fg=(len(ids)/n)*(scores['A']['subject']-scores['B']['subject']);bg=(len(other)/n)*(scores['A']['complement']-scores['B']['complement']);delta=scores['A']['all']-scores['B']['all'];residual=(fg+bg)-delta;assert abs(residual)<rule['score_reproduction_tolerance']
 e=eligibility[r['case_id']];routed=route(b,s,e['eligibility'],fg,bg,rule['epsilon'])
 records.append({'case_id':r['case_id'],'development_pair':r['pair_id'],'source_group':r['source_group'],'selection_category':r['selection_category'],'display_X_canonical':r['display_X_canonical'],'display_Y_canonical':r['display_Y_canonical'],'eligibility':e['eligibility'],'eligibility_reason':e['reason'],'subject_patch_count':len(ids),'total_source_patches':n,'subject_fraction':len(ids)/n,'scores':scores,'foreground_contribution_AminusB':fg,'complement_contribution_AminusB':bg,'global_difference_AminusB':delta,'decomposition_residual':residual,'baseline_prediction':b,'subject_prediction':s,'routed_prediction':routed['winner'],'triggered':routed['triggered'],'routing_status':routed['status'],'routing_reason':routed['reason']})
(run/'outputs/mechanism_scores_before_labels_PRIVATE.json').write_text(json.dumps({'stage':'before label read','max_cached_score_error':maxerror,'records':records},ensure_ascii=False,indent=2)+'\n')
# Labels first enter this run's computation here, after fixed rule, source QA and predictions are written.
label_df=pd.read_csv(prior/'outputs/per_pair_results.csv',usecols=['pair_id','expert_winner'],dtype=str);labels=dict(zip(label_df.pair_id,label_df.expert_winner))
credit=lambda pred,gt:.5 if pred=='tie' else float(pred==gt) if pred in ['A','B'] else 0.
counts={x:0 for x in ['rescued','new_error','unchanged','unable']}
for r in records:
 gt=labels[r['development_pair']];assert gt in ['A','B'];r['released_label']=gt;r['baseline_credit']=credit(r['baseline_prediction'],gt);r['routed_credit']=credit(r['routed_prediction'],gt);delta=r['routed_credit']-r['baseline_credit']
 outcome='rescued' if delta>0 else 'new_error' if delta<0 else 'unable' if r['routing_status']=='unable' else 'unchanged';r['outcome']=outcome;counts[outcome]+=1
assert sum(counts.values())==10
baseline=sum(r['baseline_credit'] for r in records);routed=sum(r['routed_credit'] for r in records);net=counts['rescued']-counts['new_error'];assert routed-baseline==net
summary={'run_id':run.name,'stage':'one fixed reused-development diagnostic, not independent confirmation','cases':10,'distinct_sources':len({r['source_group'] for r in records}),'disagreement_cases':6,'agreement_controls':4,'one_rule_evaluated':True,'rule_sha256':seal['rule_sha256'],'max_cached_score_reproduction_error':maxerror,'max_decomposition_residual':max(abs(r['decomposition_residual']) for r in records),'baseline_correct_credit':baseline,'routed_correct_credit':routed,'baseline_accuracy':baseline/10,'routed_accuracy':routed/10,'triggered_cases':[r['case_id'] for r in records if r['triggered']],'outcome_counts':counts,'net_correct_credit':net,'unable_kept_in_denominator':True,'control_changed_predictions':sum(r['baseline_prediction']!=r['routed_prediction'] for r in records if r['selection_category']=='agreement_control'),'cached_computation_wall_seconds':time.perf_counter()-started,'timing_scope':'cache loading, 20 candidate similarity matrices and scoring; not model inference or end-to-end latency; excludes image review','new_model_inference':False,'new_masks_or_features':False,'confirmation_access':False,'PVC_GPU_training_rentals_human_annotation':False,'decision':'STOP_no_positive_net_gain' if net<=0 else 'DIAGNOSTIC_ONLY_positive_on_selected_cases_not_independent_evidence','stop_conditions':'No threshold, source mask, batch or head retries regardless of result'}
(run/'outputs/per_case_results_PRIVATE.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
public=[]
for r in records:
 public.append({k:v for k,v in r.items() if k not in ['released_label','scores','baseline_prediction','subject_prediction','routed_prediction','display_X_canonical','display_Y_canonical']})
(run/'outputs/per_case_results_screened.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n');(run/'outputs/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2));print('CASE_SCREENED_RESULTS');print(json.dumps([{'case_id':r['case_id'],'selection_category':r['selection_category'],'eligibility':r['eligibility'],'triggered':r['triggered'],'outcome':r['outcome'],'baseline_credit':r['baseline_credit'],'routed_credit':r['routed_credit'],'foreground_contribution':r['foreground_contribution_AminusB'],'complement_contribution':r['complement_contribution_AminusB']} for r in records],ensure_ascii=False,indent=2))
