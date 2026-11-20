"""Audit existing diagnostic label lineage; no image decoding, inference or scoring."""
from pathlib import Path
import argparse,hashlib,json
import pyarrow.parquet as pq
p=argparse.ArgumentParser();p.add_argument('--experiment-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.experiment_root
relative=['data/snapshot-01/fixed_sample_manifest.json','data/snapshot-01/replication/fixed_sample_manifest.json','runs/20261120-mac-02/protocol/inputs.json','runs/20261120-mac-02/server_batch/candidate_mapping.json','runs/20261120-mac-02/server_batch/batch_manifest.json']
opened={}
def read(rel):
 raw=(root/rel).read_bytes();opened[rel]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
release={}
for rel in relative[:2]:
 for row in read(rel)['rows']:
  key=row['key']
  if key in release:assert release[key]['winner']==row['winner']
  release[key]=row
inputs=read(relative[2]);assert inputs['dataset_revision']=='4dfccf0bb4a4448aed140e8ff8ccb412e0f9b2d2'
byid={x['pair_id']:x for x in inputs['rows']};mapping=read(relative[3]);manifest=read(relative[4]);assert manifest['pairs_per_order']==16
stats={};scope_examples=[]
for order in ['ab','ba']:
 rel=f'runs/20261120-mac-02/server_batch/{order}/data/diagnostic16.parquet';path=root/rel
 # Column projection excludes embedded source and candidate image bytes.
 table=pq.ParquetFile(path).read(columns=['key','instruction','winner','task'],use_threads=False).to_pylist()
 assert len(table)==16
 m={x['pair_id']:x for x in mapping if x['order']==order};assert len(m)==16
 assert set(m)=={x['key'] for x in table}
 for row in table:
  key=row['key'];original=release[key];intermediate=byid[key]
  assert original['winner']==intermediate['expert_winner']
  expected=original['winner'] if order=='ab' else ('Image B' if original['winner']=='Image A' else 'Image A')
  assert row['winner']==expected==m[key]['ordered_expert_winner']
  assert m[key]['canonical_expert_winner']==original['winner'][-1]
  assert row['instruction']==intermediate['instruction']==original['instruction']
  assert row['task']==intermediate['task']==original['task']=='background_change'
  if order=='ab' and 'ensuring the lighting on the subjects' in row['instruction']:
   scope_examples.append({'development_pair':key,'instruction':row['instruction'],'interpretation':'The instruction explicitly allows subject-lighting integration. Pixelwise/color invariance of the entire foreground is not implied by visual-consistency labels.'})
 stats[order]={'rows':len(table),'winner_mapping_mismatches':0,'instruction_mismatches':0,'task_mismatches':0}
output={'status':'PASS','scope':'Existing development16 only; no new preference predictions or scores','dataset_revision':inputs['dataset_revision'],'unique_pairs':16,'ordered_rows_checked':32,'tasks':['background_change'],'per_order':stats,'upstream_metadata_hashes':opened,'parquet_projection':['key','instruction','winner','task'],'image_columns_read':False,'label_scope':'Released aggregated winner for visual consistency, according to official dataset card and paper; no subject-only component or individual rater ballots in released schema','instruction_scope_examples':scope_examples,'confirmation_files_opened_or_hashed':False}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print('PASS: 16 original development preferences -> 32 AB/BA records; winner/instruction/task mappings match; no images decoded or preference scores computed.')
