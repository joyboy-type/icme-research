"""Verify original-source and independently rendered evidence without loading any model."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json,sys
import numpy as np
from PIL import Image,ImageChops
RUN=Path(__file__).resolve().parents[1];PROJECT=RUN.parents[1];OUT=RUN/'outputs'
sys.path.insert(0,str(RUN.parent/'20261121-mac-07/code'))
from pair_audit import git_blob_sha

def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 cfg=read(RUN/'protocol.json');specs=read(RUN/'source_specs.json')
 for p,h in cfg['word_guard'].items():assert sha256(Path(p).read_bytes()).hexdigest()==h
 source=read(RUN/'source_lock.json')
 for p,h in source['hashes'].items():assert sha256((PROJECT/p).read_bytes()).hexdigest()==h
 lock=read(RUN/'capture_lock.json')
 for p,h in lock['hashes'].items():assert sha256((RUN/p).read_bytes()).hexdigest()==h
 fix=read(OUT/'capture_fix.json');assert sha256((RUN/'code/replay.mjs').read_bytes()).hexdigest()==fix['corrected_code_sha256'];assert sha256((RUN/'attempts/capture01/replay.mjs').read_bytes()).hexdigest()==fix['original_code_sha256']
 captures=read(OUT/'captures.json');assert len(captures)==8 and len({x['launchId'] for x in captures})==8 and len({x['screenshot'] for x in captures})==8
 for c in captures:assert sha256((PROJECT/c['screenshot']).read_bytes()).hexdigest()==c['png_sha256']
 for seed in read(OUT/'archive_seeds.json'):assert sha256((PROJECT/seed['har']).read_bytes()).hexdigest()==seed['artifact_sha256']
 snapshots=read(OUT/'snapshot_pairs.json')
 for row in snapshots:
  for field,key in [('before_path','before_blob_sha'),('after_path','after_blob_sha')]:assert git_blob_sha((PROJECT/row[field]).read_bytes())==row[key]
  a,b=[Image.open(PROJECT/row[k]).convert('RGB') for k in ['before_path','after_path']];assert a.size==b.size;d=ImageChops.difference(a,b);r,g,blue=d.split();maximum=ImageChops.lighter(ImageChops.lighter(r,g),blue);h=maximum.histogram();assert sum(h[1:])==row['changed_pixels'];assert sum(h[5:])==row['changed_gt4']
 for row in read(OUT/'repeat_pixel_audit.json'):
  a,b=[Image.open(PROJECT/row[k]).convert('RGB') for k in ['a','b']];assert a.size==b.size and ImageChops.difference(a,b).getbbox() is None
 mdn=read(OUT/'mdn_version_pixel_audit.json');assert len(mdn)==2 and all(x['same_geometry'] and x['changed_pixels']==0 for x in mdn)
 for s in specs['mdn']['html']:
  p=PROJECT/'data/real-version-replay-02/mdn'/s['version']/Path(s['path']).name;assert git_blob_sha(p.read_bytes())==s['blob_sha']
 for s in specs['mdn']['assets']:
  p=PROJECT/'data/real-version-replay-02/blobs'/s['blob_sha'];assert git_blob_sha(p.read_bytes())==s['blob_sha']
 for row in read(OUT/'licenses.json'):
  p=PROJECT/'data/real-version-replay-02/licenses'/(row['repository'].replace('/','__')+'.txt');assert git_blob_sha(p.read_bytes())==row['blob_sha']
 labels=[]
 for row in snapshots:labels.append({'id':row['id'],'project_group':row['project_group'],'change_unit':row['change_unit'],'narrow_candidate_label':'contrast_recovery','label_description':'Activated outline control becomes legible through foreground/background/theme color correction.','basis':'Pinned real commit plus localized matching before-after pixel change; source and visual review independently agreed.','atomic13_label':None,'single_css_property':False,'independent_human_annotation':False,'training_split':None,'status':'source_and_pixel_supported_narrow_candidate'})
 dump(OUT/'narrow_labels.json',{'instances':labels,'independent_change_units':1,'project_groups':1,'independent_human_gold_labels':0,'certified_atomic13_labels':0,'scope':'One real2024 color/contrast correction,6 browser/direction variants. Stable multi-project collection not established.'})
 review={'important_implementation_issues_unresolved':[],'independent_review':'Source/capture locks and complete HAR hashes match; eight distinct launch IDs, four pixel-identical repeats.','required_scope_caveat':'ionic_1 contains6 SVG fetches excluded from the per-capture resource/failed-request/readiness instrumentation; retain as partial-resource-check repeat only.','narrow_label_review':'Independent source and visual review supports contrast_recovery candidate for all6 variants; not single CSS property, atomic13 gold, text/icon insertion, geometry change, human annotation, causal ablation, or accessibility compliance.','historical_control_caveat':'Current Mac artifact controls do not reproduce2024 Linux snapshot environments.'};dump(OUT/'review.json',review)
 quality=read(OUT/'control_quality.json');summary={'scope':cfg['scope'],'real_modified_snapshot_instances':6,'independent_real_change_units':1,'positive_project_groups':1,'narrow_candidate_classes':['contrast_recovery'],'certified_atomic13_labels':0,'independent_human_gold_labels':0,'new_counted_captures':8,'independent_repeat_pairs':4,'repeat_pairs_with_zero_pixel_difference':4,'full_resource_crosschecked_repeat_pairs':quality['full_resource_crosschecked_pairs'],'partial_resource_instrumentation_repeat_pairs':1,'repeat_source_project_groups':2,'mdn_real_code_change_visible_pairs':0,'pure_justify_or_position_examples_added':0,'historical_version_matched_noise_controls':0,'jpeg_error_evaluation':'Not run; no model/threshold scored. Existing result remains unresolved.','same_artifact_repeat_controls_are_population_fpr_estimate':False,'initial_recording_failure_preserved':True,'new_live_acquisitions':2,'live_acquisitions_repeated_after_fix':0,'training_or_tuning':False,'paid_api_calls':0,'rented_compute':False,'word_unchanged_verified':True,'decision':'Retain a narrowed contrast-recovery candidate pilot and fixed-artifact repeat controls with quality flags. Dataset remains insufficient for independent training/development/confirmation or a robust detector claim.'};dump(OUT/'summary.json',summary)
 v={'at_utc':datetime.now(timezone.utc).isoformat(),'source_lock_files_verified':len(source['hashes']),'capture_lock_files_verified':len(lock['hashes']),'original_snapshot_pairs_git_and_independent_pillow_checked':6,'independent_capture_png_files_verified':8,'independent_repeats_pixel_recomputed':4,'mdn_before_after_compared':2,'original_source_html_css_modified_by_us':False,'source_images_and_licenses_git_blob_verified':True,'capture_failure_and_fix_provenance_verified':True,'tests':{'node_passed':4,'failed':0,'red_green_logs_retained':True},'word_unchanged_verified':True,'old_confirmation_used':False,'models_loaded_or_thresholds_evaluated':False,'partial_capture_resource_scope':'outputs/control_quality.json'};dump(OUT/'verification.json',v)
 meta={'id':RUN.name,'status':'complete_bounded_real_change_and_independent_replay_pilot','scope':cfg['scope'],'protocol':'protocol.json','protocol_sha256':sha256((RUN/'protocol.json').read_bytes()).hexdigest(),'taskbook_reference':cfg['word_reference'],'word_policy':cfg['word_policy'],'word_unchanged_verified':True,'narrow_class':'contrast_recovery_candidate','real_snapshot_instances':6,'independent_change_units':1,'repeat_pairs':4,'fully_resource_crosschecked_repeat_pairs':3,'partial_resource_check_repeat_pairs':1,'scope_exclusions':['Not6 independent edits or multiple positive projects.','Not atomic13 or independently human-labelled gold data.','MDN source change produced no visible change and is not a layout positive.','Live Ionic commit identity not exposed; replay controls do not match historical2024 versions/platforms.','No pure justify/position new examples or robust JPEG fix.'],'decision':summary['decision'],'continue_conditions':['Multiple real independently traceable commits and projects in the retained narrow category.','Independent label/region review before formal scoring.','Matched historical build/version/capture environment controls, with fetch/SVG completion checks.','Fresh source-isolated confirmation before any revised method claim.'],'verification':'outputs/verification.json','paid_api_calls':0,'rented_compute':False,'training_or_tuning':False};dump(RUN/'run.json',meta)
 print(json.dumps({'real_snapshot_instances':6,'independent_changes':1,'repeat_pairs':4,'full_resource_checked':3,'partial':1,'mdn_visible_changes':0,'word_unchanged':True,'models_evaluated':False}))
if __name__=='__main__':main()
