"""Reused frozen features; source review is diagnostic and not expert truth."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import argparse,json,hashlib,time,platform
import numpy as np,pandas as pd
from diagnostic_core import reviewed_support,matched_controls,match_diagnostic
R=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--archive-root',type=Path,default=R.parents[1]);args=parser.parse_args()
root=args.archive_root.resolve();old=root/'runs/20261120-mac-02';out=R/'outputs'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
annotations=load(R/'protocol/source_annotations_frozen.json')['rows'];freeze=load(R/'protocol/annotation_freeze_sha256.json')
assert sha(R/'protocol/source_annotations_frozen.json')==freeze['sha256']
meta=load(old/'outputs/source_mask_metadata_before_labels.json');audit=load(old/'outputs/input_audit_before_scoring.json')
featurepath=old/'outputs/dino_features.npz';f=np.load(featurepath);features=dict(zip(f['keys'],f['features']))
assert len(audit)==89 and len(annotations)==len(meta)==46
supports={};source_metrics=[]
for a,m in zip(annotations,meta):
 g=a['source_group'];assert g==m['source_group'];src=features[m['source_pixel_sha']]
 review=reviewed_support(a['boxes_grid_xyxy'],a['subject_eligible'])
 seed=int(hashlib.sha256(('20261120|'+g).encode()).hexdigest()[:8],16)
 controls=matched_controls(review,m['mask_patch_probabilities'],src[1:]@src[0],seed)
 assert len({len(v) for v in controls.values()})==1
 legacy=np.array(m['selected_patches'],dtype=int)
 if not len(legacy):legacy=np.array([int(np.argmax(m['mask_patch_probabilities']))])
 supports[g]={'all':np.arange(256),'auto':legacy,**controls}
 intersection=len(set(review)&set(legacy));union=len(set(review)|set(legacy))
 source_metrics.append({'source_group':g,'review_index':a['review_index'],'first_row':a['first_row'],'category':a['category'],'subject_eligible':a['subject_eligible'],
 'crop_truncated':a['crop_truncates_proposed_subject'],'legacy_empty_fallback':m['empty_fallback'],'legacy_K':len(legacy),'review_K':len(review),
 'review_box_patch_coverage_by_auto':intersection/len(review),'patch_support_iou_not_segmentation_iou':intersection/union})
save_support={g:{k:v.tolist() for k,v in data.items()} for g,data in supports.items()}
(out/'source_supports_before_scores.json').write_text(json.dumps(save_support,indent=2))
(out/'source_review_metrics.csv').write_text(pd.DataFrame(source_metrics).to_csv(index=False))
pre={'features_sha256':sha(featurepath),'annotations_sha256':freeze['sha256'],'method_sha256':sha(R/'protocol/diagnostic_frozen.json'),'source_groups':46,'pairs':89,'stage':'before_new_pair_scores','prior_feature_run':'20261120-mac-02'}
(R/'protocol/input_freeze.json').write_text(json.dumps(pre,indent=2))
annotation_by_group={a['source_group']:a for a in annotations};records=[];geometry=[];t=time.perf_counter()
for row in audit:
 g=row['source_group'];src=features[row['pixel_hashes']['source']];scores={'A':{},'B':{}}
 for candidate in ['A','B']:
  dst=features[row['pixel_hashes'][candidate]];matrix=src[1:]@dst[1:].T
  scores[candidate]['global_CLS']=float(src[0]@dst[0])
  for region,ids in supports[g].items():
   diag=match_diagnostic(matrix,ids,2)
   scores[candidate][region+'_free']=diag['score_unrestricted'];scores[candidate][region+'_r2']=diag['score_restricted']
   geometry.append({'pair_id':row['pair_id'],'source_group':g,'candidate':candidate,'region':region,'K':len(ids),**diag})
 a=annotation_by_group[g]
 records.append({'pair_id':row['pair_id'],'row_idx':row['row_idx'],'source_group':g,'category':a['category'],'subject_eligible':a['subject_eligible'],'crop_truncated':a['crop_truncates_proposed_subject'],'scores':scores})
(out/'scores_before_labels.json').write_text(json.dumps(records,indent=2))
pd.DataFrame(geometry).to_csv(out/'correspondence_geometry.csv',index=False)
# All new source supports and scores above are serialized before labels are opened.
labels={r['pair_id']:r['expert_winner'][-1] for r in load(old/'protocol/inputs.json')['rows']}
methods=list(records[0]['scores']['A']);rows=[];logs=[]
for r in records:
 gt=labels[r['pair_id']];flat={k:r[k] for k in ['pair_id','row_idx','source_group','category','subject_eligible','crop_truncated']};flat['expert_winner']=gt
 for method in methods:
  sa=r['scores']['A'][method];sb=r['scores']['B'][method];winner='A' if sa>sb else 'B' if sb>sa else 'tie';credit=.5 if winner=='tie' else float(winner==gt)
  flat[method+'_winner']=winner;flat[method+'_credit']=credit;flat[method+'_score_A']=sa;flat[method+'_score_B']=sb
  for order,ids in [('AB',['A','B']),('BA',['B','A'])]:
   x,y=[r['scores'][i][method] for i in ids];raw='Image A' if x>y else 'Image B' if y>x else 'Tie'
   mapped=ids[0] if x>y else ids[1] if y>x else 'tie';assert mapped==winner
   logs.append({'run_id':R.name,'pair_id':r['pair_id'],'source_group':r['source_group'],'method':method,'order':order,'score_a':x,'score_b':y,'raw_output':raw,'mapped_winner':mapped,'label_pref':gt,'status':'ok' if mapped!='tie' else 'tie'})
 rows.append(flat)
df=pd.DataFrame(rows);df.to_csv(out/'per_pair_results.csv',index=False);pd.DataFrame(logs).to_csv(out/'ordered_logs.csv',index=False)
legacy=pd.read_csv(old/'outputs/per_pair_results.csv').set_index('pair_id')
for new,prior in [('global_CLS','B0_DINO_CLS'),('all_free','B1_all_unrestricted'),('all_r2','A_all_radius2'),('auto_free','A_subject_unrestricted'),('auto_r2','B3_subject_radius2')]:
 assert df.set_index('pair_id')[new+'_winner'].equals(legacy.loc[df.pair_id,prior+'_winner'])
def paired(data,target,baseline):
 delta=data[target+'_credit']-data[baseline+'_credit']
 blocks=data.assign(delta=delta).groupby('source_group').delta.agg(['sum','size']).to_numpy();rng=np.random.default_rng(20261120)
 selected=blocks[rng.integers(len(blocks),size=(4000,len(blocks)))];bootstrap=selected[:,:,0].sum(axis=1)/selected[:,:,1].sum(axis=1)
 return {'target':target,'baseline':baseline,'wins':int((delta>0).sum()),'new_errors':int((delta<0).sum()),'same_credit':int((delta==0).sum()),'delta':float(delta.mean()),'source_group_95ci':np.percentile(bootstrap,[2.5,97.5]).tolist()}
comparisons=[('auto_r2','all_free'),('reviewed_free','all_free'),('reviewed_r2','all_free'),('reviewed_free','auto_free'),('reviewed_r2','auto_r2'),('reviewed_free','u2_topk_free'),('reviewed_r2','u2_topk_r2'),('reviewed_r2','reviewed_free'),('reviewed_r2','salient_topk_r2'),('reviewed_r2','random_topk_r2'),('reviewed_r2','center_topk_r2')]
subsets={}
for name,data in [('all89',df),('source_eligible',df[df.subject_eligible]),('source_fallback',df[~df.subject_eligible])]:
 subsets[name]={'pairs':len(data),'source_groups':data.source_group.nunique(),'methods':{m:{'correct':int((data[m+'_credit']==1).sum()),'ties':int((data[m+'_winner']=='tie').sum()),'accuracy':float(data[m+'_credit'].mean())} for m in methods},'paired':{a+'_vs_'+b:paired(data,a,b) for a,b in comparisons}}
sr=pd.DataFrame(source_metrics);geo=pd.DataFrame(geometry)
summary={'run_id':R.name,'stage':'reused diagnostic only; assistant boxes not expert truth','subsets':subsets,'source_review':{'groups':46,'eligible_groups':int(sr.subject_eligible.sum()),'fallback_groups':int((~sr.subject_eligible).sum()),'category_counts':sr.category.value_counts().to_dict(),'crop_truncated_flag_groups':int(sr.crop_truncated.sum()),'legacy_empty_fallback_groups':int(sr.legacy_empty_fallback.sum()),'review_box_coverage_by_auto_median_eligible':float(sr[sr.subject_eligible].review_box_patch_coverage_by_auto.median()),'review_K_median_eligible':float(sr[sr.subject_eligible].review_K.median())},'feature_match_geometry':{key:{'outside_r2_fraction_mean':float(data.outside_radius_fraction.mean()),'score_penalty_mean':float(data.score_penalty.mean())} for key,data in geo.groupby('region')},'order_inconsistency':0,'invalid_outputs':0,'elapsed_seconds':time.perf_counter()-t,'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'device':'CPU','threads':2},'caveats':['No new confirmation data scored.','Source review knows old aggregate/disagreement results; no independent expert annotation.','Boxes include background, omit thin details and do not undo center-crop loss.','Full89 reviewed results include explicit all-patch fallback on ineligible sources; eligible subset reported separately.','Feature NN displacement is not a physical correspondence annotation.','No model forward timing or PVC result.']}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
disagreements=df[df.reviewed_r2_winner!=df.all_free_winner];disagreements.to_csv(out/'reviewed_vs_B1_disagreements.csv',index=False)
print(json.dumps({'subsets':{k:{'pairs':v['pairs'],'groups':v['source_groups'],'correct':{m:z['correct'] for m,z in v['methods'].items()}} for k,v in subsets.items()},'source_review':summary['source_review'],'replay_all_predictions':'PASS','order_inconsistency':0},indent=2))
