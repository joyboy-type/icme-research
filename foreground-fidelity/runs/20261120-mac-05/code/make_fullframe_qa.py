import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import json,math
import numpy as np,pandas as pd
from PIL import Image,ImageDraw
from fullframe_core import spatial
R=Path(__file__).resolve().parents[1];old=R.parent/'20261120-mac-02';out=R/'outputs';Q=R/'qa'
meta=json.loads((old/'outputs/source_mask_metadata_before_labels.json').read_text());supports=json.loads((out/'supports_before_scores.json').read_text())
modes=['center','letterbox','square_warp']
def overlay(image,ids):
 layer=Image.new('RGBA',image.size,(0,0,0,0));d=ImageDraw.Draw(layer)
 for idx in ids:
  y,x=divmod(idx,16);d.rectangle((x*14,y*14,x*14+13,y*14+13),fill=(255,20,0,70),outline=(255,20,0,150))
 return Image.alpha_composite(image.convert('RGBA'),layer).convert('RGB')
for page in range(math.ceil(len(meta)/6)):
 view=Image.new('RGB',(1040,1560),'white');maskview=Image.new('RGB',(1040,1560),'white');dv=ImageDraw.Draw(view);dm=ImageDraw.Draw(maskview)
 for j,m in enumerate(meta[page*6:(page+1)*6]):
  y=j*260;source=Image.open(m['source_file']).convert('RGB');g=m['source_group'];thumb=source.copy();thumb.thumbnail((224,224))
  for sheet,draw in [(view,dv),(maskview,dm)]:
   sheet.paste(thumb,(2,y+28));draw.text((2,y+4),f'{page*6+j:02d} source {g[:10]}',fill='black')
  for k,mode in enumerate(modes):
   prepared,_=spatial(source,mode);x=(k+1)*260
   view.paste(prepared,(x,y+28));dv.text((x,y+4),mode,fill='black')
   maskview.paste(overlay(prepared,supports[mode][g]['auto']),(x,y+28));dm.text((x,y+4),mode+' auto K='+str(len(supports[mode][g]['auto'])),fill='black')
 view.save(Q/f'framing_{page+1}.png');maskview.save(Q/f'auto_regions_{page+1}.png')
df=pd.read_csv(out/'per_pair_results.csv');changed=df[df.letterbox_auto_r2_winner!=df.center_auto_r2_winner].sort_values('pair_id');changed.to_csv(out/'letterbox_vs_center_B3_disagreements.csv',index=False)
chosen=changed.head(12);audit={r['pair_id']:r for r in json.loads((old/'outputs/input_audit_before_scoring.json').read_text())}
for page in range(math.ceil(len(chosen)/3)):
 sheet=Image.new('RGB',(1080,780),'white');draw=ImageDraw.Draw(sheet)
 for j,(_,r) in enumerate(chosen.iloc[page*3:(page+1)*3].iterrows()):
  for k,part in enumerate(['source','A','B']):
   im=Image.open(audit[r.pair_id]['images'][part]).convert('RGB');im.thumbnail((352,224));sheet.paste(im,(k*360,j*260+28));draw.text((k*360+2,j*260+4),r.pair_id+' '+part,fill='black')
 sheet.save(Q/f'candidate_disagreements_{page+1}.png')
(out/'qa_sample.json').write_text(json.dumps({'total_B3_winner_disagreements':len(changed),'selected_pair_ids':chosen.pair_id.tolist(),'rule':'first12 lexical IDs; no labels/method names on image sheets','source_groups_in_framing_and_mask_views':46},indent=2))
print('Source views8, auto overlays8; disagreement sample',len(chosen),'of',len(changed))
