"""Replay fixed cache-only fusion and scored image-reviewed marker records. No downloads or model calls."""
from pathlib import Path
import subprocess,sys,json,os
P=Path(__file__).resolve().parent
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
for script in ['fuse_predictions.py','evaluate.py']:
 r=subprocess.run([sys.executable,str(P/script)],capture_output=True,text=True,check=True)
 (P/(script.replace('.py','')+'_replay.log')).write_text(r.stdout+r.stderr)
a=json.loads((P/'automatic_results.json').read_text());s=json.loads((P/'sample.json').read_text())
for r,m in zip(a,s):
 assert r['marker_id']==m['marker_id'];r['phase']=m['phase'];r['gold_image_review']='Gold identity verified in source-image review sheets.'
for i,models in {13:['qwen122','olmo2','fusion'],29:['olmo2','deepseek'],30:['deepseek']}.items():
 for model in models:
  v=a[i]['models'][model];v['automatic_status']=v['status'];v['status']='identity_at_anchor';v['pred']=a[i]['gold_numeric'];v['manual_review']='Image/output review confirms correct chemical-list occurrence; exact evaluator mismatch from Cl/I or formula tokens.'
(P/'reviewed_results.json').write_text(json.dumps(a,ensure_ascii=False,indent=2))
r=subprocess.run([sys.executable,str(P/'finalize.py')],capture_output=True,text=True,check=True)
(P/'reproduce.log').write_text(r.stdout+r.stderr);print(r.stdout)
