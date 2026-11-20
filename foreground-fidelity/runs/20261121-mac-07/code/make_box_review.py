"""Plot two existing AI boxes for review; never modify annotations or infer boxes."""
import argparse, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

p=argparse.ArgumentParser()
p.add_argument('--practice-root',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
original=a.practice_root/'results/deepseek-v4-flash-vision-exp.json'
d=json.loads(original.read_text());items={x['image_id']:x for x in d['image_index']};ann={x['image_id']:x for x in d['annotations']}
fig,axs=plt.subplots(2,3,figsize=(16,8),layout='constrained');records=[]
for row,ident in enumerate(['P003','P005']):
 item=items[ident];source=a.practice_root/item['file'];raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256']
 im=Image.open(source).convert('RGB');w,h=im.size;assert (w,h)==(item['width'],item['height'])
 b=ann[ident]['box_xyxy'];x1,y1,x2,y2=[b[0]*w,b[1]*h,b[2]*w,b[3]*h]
 for col in range(3):
  ax=axs[row,col];ax.imshow(im,extent=(0,1,1,0));ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='red',lw=1.4));ax.grid(alpha=.2);ax.set_xlabel('x / source width');ax.set_ylabel('y / source height');ax.set_aspect('auto')
 axs[row,0].set_title(ident+' original + submitted box')
 axs[row,1].set_xlim(.18,.83);axs[row,1].set_ylim(min(1,b[3]+.08),max(0,b[3]-.13));axs[row,1].set_title('Lower edge / visible wheels')
 axs[row,2].set_xlim(.2,.9);axs[row,2].set_ylim(min(1,b[1]+.1),max(0,b[1]-.1));axs[row,2].set_title('Upper edge / roof accessories')
 records.append({'image_id':ident,'source_sha256':item['sha256'],'dimensions':[w,h],'original_box_xyxy':b,'pixel_edges_float':[x1,y1,x2,y2],'geometry_changed':False})
fig.suptitle('AI-assisted inspection only; red = original submitted box; no corrected boxes or human truth')
fig.savefig(a.output_dir/'two_vehicle_box_review.png',dpi=140);plt.close(fig)
(a.output_dir/'box_geometry_reference.json').write_text(json.dumps({'annotation_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'records':records,'scope':'Pure display of existing coordinates; no inference, new boxes or preference scores'},indent=2))
print('Rendered original boxes with coordinate axes; source hashes verified; original records unchanged.')
