"""Execute the pinned official prepare_dataset method without GPU imports."""
from pathlib import Path
from types import SimpleNamespace
from typing import Dict
import ast,json,hashlib,importlib.metadata as md
from PIL import Image
R=Path(__file__).resolve().parents[1];previous=R.parent/'20261120-mac-02'
source=previous/'protocol/official_pvc/src__common_utils__dataset_loader.py'
tree=ast.parse(source.read_text())
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='VCRewardBenchmark')
method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='prepare_dataset')
module=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(module)
namespace={'Dict':Dict};exec(compile(module,str(source),'exec'),namespace)
loaded={order:namespace['prepare_dataset'](SimpleNamespace(bench_path=str((previous/'server_batch'/order).resolve()))) for order in ['ab','ba']}
mapping=json.loads((previous/'server_batch/candidate_mapping.json').read_text())
assert len({m['source_group'] for m in mapping})==16
assert len(loaded['ab'])==len(loaded['ba'])==16
assert set(loaded['ab'])==set(loaded['ba'])
for key,a in loaded['ab'].items():
 b=loaded['ba'][key]
 assert a['instruction']==b['instruction'] and a['task_type']==b['task_type']=='background_change'
 assert a['winner']!=b['winner']
 for item in [a,b]:
  assert isinstance(item['input_image'],Image.Image)
  assert len(item['edited_images'])==2
  for im in [item['input_image'],*item['edited_images']]:
   assert isinstance(im,Image.Image);im.load()
 assert a['input_image'].convert('RGB').tobytes()==b['input_image'].convert('RGB').tobytes()
 assert a['edited_images'][0].convert('RGB').tobytes()==b['edited_images'][1].convert('RGB').tobytes()
 assert a['edited_images'][1].convert('RGB').tobytes()==b['edited_images'][0].convert('RGB').tobytes()
result={'status':'PASS','official_commit':'a083bbbf28f86c6e1f9e50c11082b80fccbe4ceb','official_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'execution':'Exact AST-extracted VCRewardBenchmark.prepare_dataset method; module GPU/unrelated imports not executed','pairs_per_order':16,'source_groups':16,'ordered_inputs':32,'image_decodes':96,'candidate_and_label_swap':'PASS','dependencies':{k:md.version(k) for k in ['datasets','pyarrow','Pillow','multiprocess','dill','xxhash','fsspec','huggingface-hub']},'device':'CPU','thread_limit':2,'new_predictions':0,'model_weights_loaded':False,'GPU_used':False,'paid_resources_used':False,'limitations':'Data loading verified locally; full official dependency environment, model loading and GPU inference remain untested.'}
(R/'outputs/official_loader_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
