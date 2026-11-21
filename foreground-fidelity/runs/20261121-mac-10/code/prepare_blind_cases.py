"""Fixed cached-prediction sampling and blinded display; never load label/score columns."""
import argparse,hashlib,json,textwrap
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--experiment-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();root=a.experiment_root.resolve();run=a.run_root.resolve()
config=json.loads((run/'protocol/selection_frozen.json').read_text());seed=config['ranking_seed'];rank=lambda value:hashlib.sha256((seed+'|'+value).encode()).hexdigest()
prior=root/'runs/20261120-mac-02';csv=prior/'outputs/per_pair_results.csv';columns=['pair_id','source_group','B1_all_unrestricted_winner','A_subject_unrestricted_winner'];df=pd.read_csv(csv,usecols=columns,dtype=str)
assert len(df)==89 and df.source_group.nunique()==46
metadata=json.loads((prior/'outputs/input_audit_before_scoring.json').read_text());byid={r['pair_id']:r for r in metadata};assert all('expert_winner' not in r for r in metadata)
selected=[];used=set()
for disagreement,limit in [(True,6),(False,4)]:
 pool=[r for r in df.to_dict('records') if r[columns[2]] in ['A','B'] and r[columns[3]] in ['A','B'] and (r[columns[2]]!=r[columns[3]])==disagreement]
 for row in sorted(pool,key=lambda r:rank('pick|'+r['pair_id'])):
  if row['source_group'] in used:continue
  used.add(row['source_group']);selected.append({**row,'selection_category':'disagreement' if disagreement else 'agreement_control'})
  if sum(x['selection_category']==('disagreement' if disagreement else 'agreement_control') for x in selected)==limit:break
selected.sort(key=lambda r:rank('display|'+r['pair_id']))
public=[];private=[]
for i,row in enumerate(selected,1):
 case=f'C{i:03d}';r=byid[row['pair_id']];swapped=int(rank('swap|'+row['pair_id'])[-1],16)%2==1;order=['B','A'] if swapped else ['A','B']
 item={'case_id':case,'instruction':r['instruction'],'display_original':r['images']['source'],'display_X':r['images'][order[0]],'display_Y':r['images'][order[1]]};public.append(item)
 private.append({**row,'case_id':case,'display_X_canonical':order[0],'display_Y_canonical':order[1],'metadata':r})
(run/'protocol/selection_PRIVATE.json').write_text(json.dumps({'read_csv_columns':columns,'pool_rows':89,'selected':private,'actual_counts':{cat:sum(x['selection_category']==cat for x in selected) for cat in ['disagreement','agreement_control']}},ensure_ascii=False,indent=2)+'\n')
(run/'protocol/blind_review_manifest_PRIVATE.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n')
for page in range((len(public)+1)//2):
 cases=public[page*2:page*2+2];fig,axs=plt.subplots(len(cases),3,figsize=(18,5.8*len(cases)),squeeze=False)
 for j,item in enumerate(cases):
  for k,label in enumerate(['Original','X','Y']):
   key='display_original' if k==0 else 'display_'+label;path=Path(item[key]);raw=path.read_bytes();m=private[page*2+j]['metadata'];part='source' if k==0 else private[page*2+j]['display_'+label+'_canonical'];assert hashlib.sha256(raw).hexdigest()==m['file_hashes'][part]
   with Image.open(path) as im:axs[j,k].imshow(im.convert('RGB'))
   axs[j,k].axis('off');axs[j,k].set_title(item['case_id']+' / '+label)
  instruction='\n'.join(textwrap.wrap(item['instruction'],width=135));y=1-j/len(cases)-.025;fig.text(.02,y,instruction,fontsize=11,va='top')
 fig.subplots_adjust(top=.91,bottom=.03,hspace=.35,wspace=.04);fig.savefig(run/'qa'/f'blind_cases_{page+1}.png',dpi=120);plt.close(fig)
summary={'selection_frozen_sha256':hashlib.sha256((run/'protocol/selection_frozen.json').read_bytes()).hexdigest(),'selected_cases':len(public),'distinct_sources':len(used),'disagreements':sum(x['selection_category']=='disagreement' for x in selected),'agreement_controls':sum(x['selection_category']=='agreement_control' for x in selected),'labels_or_numeric_scores_loaded':False,'per_pair_winners_hidden_from_reviewer':True,'sheets':(len(public)+1)//2}
(run/'outputs/selection_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
