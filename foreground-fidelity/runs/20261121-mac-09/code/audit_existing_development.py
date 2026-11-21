"""Check existing unsealed development assets only; no scoring or resource probing."""
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resource-history-ref',required=True);a=p.parse_args();workspace=a.workspace.resolve();base=workspace/'experiments/foreground-fidelity';repo=workspace/'icme-research'
read_paths=[]
def load(path):
 read_paths.append(str(path.relative_to(workspace)));return json.loads(path.read_text())
known={x['pair_id'] for x in load(base/'runs/20261120-mac-02/protocol/inputs.json')['rows']};assert len(known)==89
old=load(base/'runs/20261120-mac-04/protocol/source_review_manifest.json');groups={x['source_group']:x for x in old};assert len(groups)==46
inventory=[];records=[]
for split in [base/'data/snapshot-01',base/'data/snapshot-01/replication']:
 rows=load(split/'fixed_sample_manifest.json')['rows'];complete=[]
 for row in rows:
  idx=row['row_idx'];key=row['key'];assert 0<=idx<100, 'Only already-used first100 metadata; never access reserved rows'
  paths=[split/'images'/f'{idx:04d}_{name}.jpg' for name in ['source','A','B']]
  if not all(x.is_file() for x in paths):continue
  complete.append(key)
  if key in known:continue
  # Open only source originals; candidate A/B existence is checked without decoding.
  with Image.open(paths[0]) as im:rgb=im.convert('RGB');pixel=hashlib.sha256(rgb.tobytes()).hexdigest();dims=list(rgb.size)
  read_paths.append(str(paths[0].relative_to(workspace)));match=groups.get(pixel)
  records.append({'development_pair':key,'row_index':idx,'source_RGB_sha256':pixel,'dimensions':dims,'matches_old46_source_group':match is not None,'old_source_first_row':match['first_row'] if match else None})
 inventory.append({'metadata_rows':len(rows),'complete_triplets':len(complete),'complete_pairs_outside_old89':len(set(complete)-known)})
exclusions=load(workspace/'work/image_preference_round3_20261120/replication/source_overlap_exclusions.json')
assert sorted(x['row_index'] for x in records)==sorted(exclusions)
assert len(records)==11 and all(x['matches_old46_source_group'] for x in records)
f=workspace/'work/image_preference_round3_20261120/edithf_test.csv';read_paths.append(str(f.relative_to(workspace)))
with f.open() as handle:
 reader=csv.DictReader(handle);columns=reader.fieldnames;count=sum(1 for _ in reader)
handoff=base/'runs/20261120-mac-06/outputs/server_handoff_current.json';index=load(base/'integrity.json');old_hash=next(x['sha256'] for x in index if x['path']=='runs/20261120-mac-06/outputs/server_handoff_current.json');current_hash=hashlib.sha256(handoff.read_bytes()).hexdigest();read_paths.append(str(handoff.relative_to(workspace)))
assert current_hash==old_hash
changes=subprocess.check_output(['git','log','--oneline','a8eb73ee36b4a63b3a40bb4abadeabdcec5ea8d2..'+a.resource_history_ref,'--','foreground-fidelity'],cwd=repo,text=True).strip()
assert not changes, 'New direction commit needs inspection before reusing resource status'
result={'status':'PASS_existing_asset_independence_audit','run_id':'20261121-mac-09','existing_scored_pairs':89,'existing_source_groups':46,'existing_unsealed_metadata_rows':100,'triplet_inventory':inventory,'extra_complete_pairs':11,'extra_pair_distinct_sources':len({x['source_RGB_sha256'] for x in records}),'extra_pairs_matching_old_sources':11,'eligible_independent_new_source_groups':0,'extra_pair_records':records,'matches_prior_overlap_exclusion_list':True,'EditHF_local_assets':{'score_rows':count,'columns':columns,'complete_original_instruction_AB_bundle_established':False,'local_image_archives_in_known_work_directory':len(list(f.parent.glob('*.7z*'))),'limitation':'Scores table and remote archive listings do not constitute local traceable image triplets or source-disjoint failure cases'},'resource_delta':{'resource_history_ref':a.resource_history_ref,'direction_commits_since_mac08':0,'known_server_handoff_hash_unchanged':True,'new_authorized_server_access_supplied':False,'hardware_or_model_cache_preflight_repeated':False,'reuse_preflight':'runs/20261121-mac-08/outputs/resource_preflight.json'},'new_preference_predictions_or_scores':0,'traceable_independent_failure_case_established':False,'confirmation_files_opened_or_hashed':False,'GPU_model_inference':False,'downloads':0,'read_scope':read_paths,'scope_note':'Only existing known local assets; not a claim that all public independent datasets are unavailable'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','extra_complete_pairs','extra_pair_distinct_sources','extra_pairs_matching_old_sources','eligible_independent_new_source_groups','new_preference_predictions_or_scores']},ensure_ascii=False,indent=2))
