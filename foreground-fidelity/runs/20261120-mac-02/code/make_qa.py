from pathlib import Path
import json,math
from PIL import Image,ImageDraw,ImageFont
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];OUT=R/'outputs';QA=R/'qa'
meta=json.loads((OUT/'source_mask_metadata_before_labels.json').read_text())
for page in range(math.ceil(len(meta)/24)):
 batch=meta[page*24:(page+1)*24];sheet=Image.new('RGB',(1800,math.ceil(len(batch)/6)*200),'white');d=ImageDraw.Draw(sheet)
 for i,r in enumerate(batch):
  src=Image.open(r['source_file']).convert('RGB');mask=Image.open(OUT/'source_masks'/f"{r['source_group']}.png").convert('L')
  overlay=Image.blend(src,Image.composite(Image.new('RGB',src.size,(255,30,50)),src,mask),.45)
  for c,im in enumerate([src,overlay]):
   im.thumbnail((148,165));sheet.paste(im,((i%6)*300+c*150,(i//6)*200+24))
  d.text(((i%6)*300+3,(i//6)*200+4),f"{r['source_group'][:8]} area {r['area_fraction_at_dino_crop']:.2f} empty {r['empty_fallback']}",fill='black')
 sheet.save(QA/f'masks_{page+1}.png')
rows={r['pair_id']:r for r in json.loads((R/'protocol/inputs.json').read_text())['rows']};df=pd.read_csv(OUT/'per_pair_results.csv');changed=df[df.B1_all_unrestricted_winner!=df.B3_subject_radius2_winner].sort_values('pair_id')
blinded=[]
for i,r in changed.iterrows():
 item=rows[r.pair_id];blinded.append({'review_id':r.pair_id,'instruction':item['instruction'],'images':item['images']})
# No method names, predictions or human winners on these review sheets.
for page in range(math.ceil(len(blinded)/4)):
 batch=blinded[page*4:(page+1)*4];sheet=Image.new('RGB',(960,len(batch)*240),'white');d=ImageDraw.Draw(sheet)
 for j,r in enumerate(batch):
  for c,part in enumerate(['source','A','B']):
   im=Image.open(r['images'][part]).convert('RGB');im.thumbnail((314,208));sheet.paste(im,(c*320,j*240+26));d.text((c*320+3,j*240+5),r['review_id']+' '+part,fill='black')
 sheet.save(QA/f'blinded_disagreements_{page+1}.png')
(OUT/'blinded_review_manifest.json').write_text(json.dumps(blinded,ensure_ascii=False,indent=2));changed.to_csv(OUT/'B3_vs_B1_disagreements.csv',index=False)
print('mask sheets',math.ceil(len(meta)/24),'disagreement pairs',len(blinded),'empty sources',[r['source_group'] for r in meta if r['empty_fallback']])
