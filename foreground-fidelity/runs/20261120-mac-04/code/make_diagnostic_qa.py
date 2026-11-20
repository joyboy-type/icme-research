from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import json,math
import pandas as pd
from PIL import Image,ImageDraw
from torchvision.transforms import functional as F,InterpolationMode
R=Path(__file__).resolve().parents[1];old=R.parent/'20261120-mac-02';Q=R/'qa'
annotations=json.loads((R/'protocol/source_annotations_frozen.json').read_text())['rows']
supports=json.loads((R/'outputs/source_supports_before_scores.json').read_text())
def mark(crop,ids,color):
 im=crop.convert('RGBA');layer=Image.new('RGBA',im.size,(0,0,0,0));d=ImageDraw.Draw(layer)
 for idx in ids:
  y,x=divmod(idx,16);d.rectangle((14*x,14*y,14*x+13,14*y+13),fill=(*color,80),outline=(*color,180))
 return Image.alpha_composite(im,layer).convert('RGB')
for page in range(math.ceil(len(annotations)/6)):
 sheet=Image.new('RGB',(1400,840),'white');d=ImageDraw.Draw(sheet)
 for j,a in enumerate(annotations[page*6:(page+1)*6]):
  x=j%2*700;y=j//2*280;g=a['source_group']
  im=Image.open(a['source_file']).convert('RGB');crop=F.center_crop(F.resize(im,224,InterpolationMode.BICUBIC),[224,224])
  d.text((x+2,y+2),f"{a['review_index']:02d} row{a['first_row']} {a['category']} crop-cut:{a['crop_truncates_proposed_subject']}",fill='black')
  for k,(label,pic) in enumerate([('source crop',crop),('legacy U2 mask',mark(crop,supports[g]['auto'],(0,170,0))),('review box union',mark(crop,supports[g]['reviewed'],(220,20,0)))]):
   d.text((x+k*230+2,y+23),label,fill='black');sheet.paste(pic,(x+k*230+2,y+42))
 sheet.save(Q/f'region_comparison_{page+1}.png')
df=pd.read_csv(R/'outputs/reviewed_vs_B1_disagreements.csv');audit={r['pair_id']:r for r in json.loads((old/'outputs/input_audit_before_scoring.json').read_text())}
for page in range(math.ceil(len(df)/3)):
 sheet=Image.new('RGB',(1080,780),'white');d=ImageDraw.Draw(sheet)
 for j,(_,row) in enumerate(df.iloc[page*3:(page+1)*3].iterrows()):
  r=audit[row.pair_id]
  for k,part in enumerate(['source','A','B']):
   im=Image.open(r['images'][part]).convert('RGB');im.thumbnail((352,224));sheet.paste(im,(k*360,j*260+28));d.text((k*360+2,j*260+5),r['pair_id']+' '+part,fill='black')
 sheet.save(Q/f'disagreements_blinded_{page+1}.png')
print('Generated8 region comparison sheets and3 disagreement sheets; no labels or methods on disagreement sheets.')
