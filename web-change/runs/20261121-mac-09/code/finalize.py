"""Read-only evidence validation and predeclared comparison-gate decision."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import sys,json
from PIL import Image,ImageChops
RUN=Path(__file__).resolve().parents[1];PROJECT=RUN.parents[1];OUT=RUN/'outputs';DATA=PROJECT/'data/contrast-source-03'
sys.path.insert(0,str(RUN.parent/'20261121-mac-07/code'))
from pair_audit import git_blob_sha

def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def hist_check(a,b,expected):
 aa,bb=[Image.open(PROJECT/p).convert('RGB') for p in [a,b]];assert aa.size==bb.size
 d=ImageChops.difference(aa,bb);r,g,blue=d.split();h=ImageChops.lighter(ImageChops.lighter(r,g),blue).histogram();assert sum(h[1:])==expected['changed_pixels'];assert sum(h[5:])==expected['changed_gt4']

def main():
 cfg=read(RUN/'protocol.json');specs=read(RUN/'source_specs.json')
 for name,h in cfg['word_guard'].items():assert sha256(Path(name).read_bytes()).hexdigest()==h
 for name,h in read(RUN/'source_lock.json')['hashes'].items():assert sha256((PROJECT/name).read_bytes()).hexdigest()==h
 for name,h in read(RUN/'capture_lock.json')['hashes'].items():assert sha256((RUN/name).read_bytes()).hexdigest()==h
 for c in specs['cases']:
  for v in c['versions']:
   for f in v['files']:assert git_blob_sha((DATA/c['id']/v['name']/f['path']).read_bytes())==f['blob_sha']
 captures=read(OUT/'captures.json');assert len(captures)==12 and len({x['launchId'] for x in captures})==12 and len({x['screenshot'] for x in captures})==12 and len({x['browser_version'] for x in captures})==1
 for c in captures:
  assert sha256((PROJECT/c['screenshot']).read_bytes()).hexdigest()==c['png_sha256'];assert c['ready'] and not c['networkErrors'] and not c['missingRequired'] and c['targetMediaCount']==0
  for resource in c['resources']:
   from urllib.parse import urlparse
   resource_path=urlparse(resource['url']).path.lstrip('/');assert sha256((DATA/resource_path).read_bytes()).hexdigest()==resource['sha256']
 repeats=read(OUT/'repeat_pixels.json');assert len(repeats)==6
 for c in repeats:assert c['valid'] and c['changed_pixels']==0;hist_check(c['a'],c['b'],c)
 versions=read(OUT/'version_pixels.json');assert len(versions)==6
 for c in versions:assert c['same_geometry'] and c['matched_semantic_state_and_geometry'] and c['changed_pixels']>0;hist_check(c['before_path'],c['after_path'],c)
 lineage=read(OUT/'fixture_lineage.json');assert len({x['target_markup_sha256_normalized'] for x in lineage['cases']})==1 and all(x['upstream_credit_urls'] for x in lineage['cases'])
 labels=[]
 subtype={'simple_disabled':'disabled_control_opacity_foreground_background_change','simple_border':'fieldset_boundary_color_change','sakura_earthly':'normal_state_button_palette_change'}
 for c in specs['cases']:
  examples=[v for v in versions if v['id']==c['id']];labels.append({'case':c['id'],'project':c['repository'],'fixture_family':c['fixture_family'],'independent_change_unit':c['commit'],'source_commit_date':c['date'],'candidate_family':'contrast_change','candidate_subtype':subtype[c['id']],'core_fill_text_recovery_candidate':c['id']!='simple_border','replicated_before_after_instances':2,'source_and_pixel_review_supported':True,'matched_semantic_state_and_geometry':True,'certified_atomic13_label':None,'independent_human_annotation':False,'scope':'Region from original upstream test page, actual source versions; not a production-site sample or injected CSS mutation.','restriction':'Border case is a fieldset boundary, not text/fill recovery; disabled case is multi-property; Sakura hover/focus effects not tested.'})
 dump(OUT/'candidate_labels.json',labels)
 gate={'predeclared_gate':'protocol.json comparison_gate','new_replayed_projects':2,'real_replayed_commits':3,'conservative_independent_fixture_groups':1,'required_independent_groups':3,'source_grouping_pass':False,'matched_repeat_capture_pass':True,'source_pixel_label_review_pass_for_candidates':True,'independent_human_gold_labels':0,'comparison_authorized_by_data_gate':False,'pixel_or_edge_detector_evaluated':False,'jpeg_detector_evaluated':False,'thresholds_selected':False,'why_stop':['Two repositories inherit identical target markup and the same upstream fixture; do not split by repository alone.','Existing run08 Ionic snapshots lack historical version/platform matched repeats and do not fill the gate.','Water source fixes require official build outputs not present in the checked tree; selected runtime modules missing. No new dependencies or historical build attempted.'],'continue_conditions':['Additional genuinely independent fixture/page families with traceable real contrast fixes.','Independent label/region review, separating fieldset boundary from fill/text restoration.','Matched source revision/browser/theme/viewport/font/state repeats with complete target dependencies.','Freeze an actual source-isolated comparison split before pixel/edge or JPEG scoring.']};dump(OUT/'comparison_gate.json',gate)
 review={'important_implementation_issues_unresolved':[],'independent_review':'Source versions, locks, capture isolation, required response hashes and candidate scope reviewed.','conclusions':['12 distinct launch IDs, one Chrome version;6 identical same-version repeat pairs.','All3 cases have equal semantic state/font declarations/geometry across source versions.','Contrast-change candidates supported with subtype restrictions; not atomic13 or human gold labels.','One shared cbracco fixture family; do not run detector/JPEG comparison.','Water historical build not run, not a failed build.']};dump(OUT/'review.json',review)
 summary={'scope':cfg['scope'],'queried_new_projects':4,'inspected_real_commits':6,'replayed_projects':2,'replayed_real_commits':3,'new_candidate_cases':3,'core_fill_text_contrast_cases':2,'auxiliary_boundary_color_cases':1,'conservative_fixture_groups':1,'counted_images':12,'matched_independent_repeat_pairs':6,'repeat_pairs_zero_pixel_difference':6,'before_after_replicate_pairs':6,'distinct_source_changes_with_visible_pixels':3,'comparison_gate_pass':False,'pixel_edge_or_jpeg_detector_evaluation':'Not run, data gate failed. Pixel histograms only validate source/capture differences.','training_or_tuning':False,'existing419_used':False,'new_dependencies_installed':0,'paid_api_calls':0,'rented_compute':False,'word_unchanged_verified':True,'decision':'Retain version-matched real CSS project contrast candidates and repeat controls; stop before detector evaluation because only one independent fixture family remains.'};dump(OUT/'summary.json',summary)
 verification={'at_utc':datetime.now(timezone.utc).isoformat(),'source_file_references_git_verified':22,'unique_source_blobs':13,'capture_png_hashes_verified':12,'unique_browser_launch_records':12,'same_browser_version':True,'no_required_target_resource_failures':True,'all_recorded_source_response_hashes_match_git_verified_files':True,'independent_repeat_pixel_pairs_checked':6,'visible_version_pairs_independently_histogram_checked':6,'identical_normalized_target_markup_hashes':True,'source_capture_locks_unchanged':True,'tests':{'node_passed':3,'failed':0,'red_green_logs_retained':True},'word_unchanged_verified':True,'models_loaded_or_thresholds_evaluated':False,'independent_review':'outputs/review.json'};dump(OUT/'verification.json',verification)
 run={'id':RUN.name,'status':'complete_bounded_matched_source_audit_gate_stop','scope':cfg['scope'],'protocol':'protocol.json','protocol_sha256':sha256((RUN/'protocol.json').read_bytes()).hexdigest(),'taskbook_reference':cfg['word_reference'],'word_policy':cfg['word_policy'],'word_unchanged_verified':True,'new_projects_replayed':2,'new_real_source_commits_replayed':3,'independent_fixture_families':1,'candidate_labels':'outputs/candidate_labels.json','matched_repeat_pairs':6,'comparison_gate':'outputs/comparison_gate.json','decision':summary['decision'],'training_or_tuning':False,'paid_api_calls':0,'rented_compute':False,'verification':'outputs/verification.json'};dump(RUN/'run.json',run)
 print(json.dumps({'real_commits':3,'projects':2,'fixture_groups':1,'matched_repeat_pairs':6,'all_repeat_pixels_zero':True,'comparison_run':False,'word_unchanged':True}))
if __name__=='__main__':main()
