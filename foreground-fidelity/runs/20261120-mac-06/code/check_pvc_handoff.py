"""Verify package and generated CLI without importing or launching judge models."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import json,hashlib,zipfile,io,tempfile,subprocess,sys,ast,typing
import pyarrow.parquet as pq,yaml
from PIL import Image
R=Path(__file__).resolve().parents[1];root=R.parents[1];old=root/'runs/20261120-mac-02';sha=lambda b:hashlib.sha256(b).hexdigest()
archive=root/'deliveries/PVC-Judge_16对双顺序输入_20261120.zip';receipt=json.loads(archive.with_suffix('.json').read_text());assert sha(archive.read_bytes())==receipt['sha256']
prefix='PVC-Judge_16对双顺序输入/'
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None;manifest=json.loads(z.read(prefix+'package_manifest.json'))
 for e in manifest['files']:assert sha(z.read(prefix+e['path']))==e['sha256']
 rows={order:pq.read_table(io.BytesIO(z.read(prefix+f'server_batch/{order}/data/diagnostic16.parquet'))).to_pylist() for order in ['ab','ba']}
 for order in ['ab','ba']:
  assert sha(z.read(prefix+f'server_batch/{order}/data/diagnostic16.parquet'))==sha((old/f'server_batch/{order}/data/diagnostic16.parquet').read_bytes())
 configure=z.read(prefix+'code/configure_pvc_server.py');assert configure==(old/'code/configure_pvc_server.py').read_bytes()
 mapping=json.loads(z.read(prefix+'server_batch/candidate_mapping.json'))
 assert len(mapping)==32 and len({m['source_group'] for m in mapping})==16
assert len(rows['ab'])==len(rows['ba'])==16
for a,b in zip(rows['ab'],rows['ba']):
 assert a['key']==b['key'] and a['instruction']==b['instruction'] and a['source_image']==b['source_image']
 assert a['edited_images']==b['edited_images'][::-1] and a['winner']!=b['winner']
 for row in [a,b]:
  for im in [row['source_image'],*row['edited_images']]:
   with Image.open(io.BytesIO(im['bytes'])) as image:image.load()
official=old/'protocol/official_pvc/src__inference__run_eval.py';tree=ast.parse(official.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args')
ns={'CONFIGS_ROOT':Path('/server/configs'),'OUTPUT_ROOT':Path('/server/outputs'),'CACHE_ROOT':'/server/cache'};exec(compile(ast.Module(body=[function],type_ignores=[]),str(official),'exec'),ns)
with tempfile.TemporaryDirectory(prefix='pvc handoff check ') as temp:
 tmp=Path(temp).resolve();batch=tmp/'batch with spaces';batch.mkdir();generator=tmp/'configure.py';generator.write_bytes(configure)
 command=[sys.executable,str(generator),'--batch-root',str(batch),'--repo-root','/server/repo with spaces/GEditBench_v2','--base-model','/server/models/Qwen3-VL-8B-Instruct','--lora','/server/models/PVC-Judge/lora_weights']
 result=subprocess.run(command,capture_output=True,text=True);assert result.returncode==0,result.stderr
 script=batch/'commands_to_review.sh';assert subprocess.run(['bash','-n',str(script)],capture_output=True).returncode==0
 fake=tmp/'fakebin';fake.mkdir();rec=tmp/'argv.jsonl'
 stub=fake/'python';stub.write_text("#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ['PVC_ARG_RECORD'],'a') as f:f.write(json.dumps({'argv':sys.argv[1:],'cwd':os.getcwd(),'PYTHONPATH':os.environ.get('PYTHONPATH')})+'\\n')\n");stub.chmod(0o755)
 env={**os.environ,'PATH':str(fake)+os.pathsep+os.environ['PATH'],'PVC_ARG_RECORD':str(rec)}
 dry=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True);assert dry.returncode==0,dry.stderr
 calls=[json.loads(line) for line in rec.read_text().splitlines()];assert len(calls)==2
 args=[]
 for call,order in zip(calls,['ab','ba']):
  argv_before=sys.argv;sys.argv=['run_eval.py',*call['argv'][1:]]
  try:parsed=ns['parse_args']()
  finally:sys.argv=argv_before
  assert parsed.bmk=='vc_reward' and parsed.seed==42 and parsed.temperature==.7 and parsed.num_pass==1
  assert parsed.prompt_id=='vlm/assessment/visual_consistency/pairwise' and parsed.prompt_version=='v1'
  assert parsed.image_min_pixels==256*32*32 and parsed.image_max_pixels==1280*32*32
  assert parsed.use_vllm and parsed.use_flash_attn and parsed.do_sample and parsed.save_details
  assert parsed.output_dir==str(batch/f'run_{order}/outputs')
  config=json.loads(Path(parsed.bmk_config).read_text());assert config['vc_reward']['bench_path']==str(batch/order)
  assert call['cwd']==str(batch/f'run_{order}') and call['PYTHONPATH']=='/server/repo with spaces/GEditBench_v2:/server/repo with spaces/GEditBench_v2/src'
  args.append(vars(parsed))
 assert args[0]['output_dir']!=args[1]['output_dir']
 (R/'outputs/server_cli_argument_only_dry_run.json').write_text(json.dumps({'calls':calls,'parsed':args,'mode':'python executable replaced by argv recorder; no judge script executed'},indent=2))
 (R/'protocol/validated_command_example_NOT_EXECUTED.sh').write_text('# Example uses deleted temporary test paths; regenerate on the actual server.\n'+script.read_text())
prompt_source=R/'protocol/official_prompt_manager.py';ptree=ast.parse(prompt_source.read_text());cls=next(n for n in ptree.body if isinstance(n,ast.ClassDef) and n.name=='PromptAssetStore');cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['__init__','_get_file_path']]
globals_={'__file__':'/server/GEditBench_v2/src/prompts/prompt_manager.py','Path':Path,'Dict':typing.Dict,'PromptTemplate':object,'os':os}
exec(compile(ast.Module(body=[cls],type_ignores=[]),str(prompt_source),'exec'),globals_)
path=globals_['PromptAssetStore']('assets')._get_file_path('vlm/assessment/visual_consistency/pairwise','v1')
assert str(path)=='/server/GEditBench_v2/src/prompts/assets/vlm/assessment/visual_consistency/pairwise/v1.yaml'
prompt=yaml.safe_load((old/'protocol/official_pvc/pairwise_v1.yaml').read_text());assert prompt['prompt_id']=='vlm/assessment/visual_consistency/pairwise' and prompt['version']=='v1'
loader=json.loads((root/'runs/20261120-mac-03/outputs/official_loader_check.json').read_text());assert loader['status']=='PASS' and loader['ordered_inputs']==32
result={'status':'local_handoff_checks_PASS_inference_NOT_RUN','package_sha256':receipt['sha256'],'pairs_per_order':16,'source_groups':16,'orders':2,'expected_inference_requests':32,'image_decodes':96,'package_integrity':'PASS','AB_BA_images_and_labels_swap':'PASS','official_data_loader':'Prior mac-03 PASS; package Parquet bytes verified identical in this run','official_argparse':'PASS for both generated commands','shell_syntax':'PASS','spaces_and_PYTHONPATH':'PASS in argument-only recorder dry run','separate_output_directories':'PASS','prompt_path_independent_of_working_directory':'PASS via exact official path methods','real_model_or_script_launched':False,'GPU_or_paid_resources_used':False,'dependencies_remaining':['actual accessible GPU server and login','pinned official checkout and complete compatible CUDA/vLLM/flash-attn environment','actual Qwen3-VL-8B-Instruct base model and PVC LoRA paths and hashes','server writable cache/output paths and measured VRAM/latency','real32 inference attempts and raw outputs including missing/invalid cases'],'correction':'Old mac-02 handoff text predates successful local datasets loader check. Local data loading is tested; full server environment and inference are not.'}
(R/'outputs/PVC_handoff_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
