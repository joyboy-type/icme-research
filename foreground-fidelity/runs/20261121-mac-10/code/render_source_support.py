"""Display existing source-only supports and their fixed crop, without pair scores."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
p=argparse.ArgumentParser();p.add_argument('--experiment-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();run=a.run_root.resolve();prior=a.experiment_root.resolve()/'runs/20261120-mac-02'
cases=json.loads((run/'protocol/selection_PRIVATE.json').read_text())['selected'];supports=json.loads((prior/'outputs/supports_before_labels.json').read_text())
for page in range((len(cases)+1)//2):
 subset=cases[2*page:2*page+2];fig,axs=plt.subplots(len(subset),3,figsize=(15,5.3*len(subset)),squeeze=False)
 for j,r in enumerate(subset):
  with Image.open(r['metadata']['images']['source']) as im:im=im.convert('RGB');w,h=im.size
  size=(int(w*224/h),224) if w>=h else (224,int(h*224/w));resized=im.resize(size,Image.Resampling.BICUBIC);left=int(round((size[0]-224)/2));top=int(round((size[1]-224)/2));crop=resized.crop((left,top,left+224,top+224))
  ax=axs[j,0];ax.imshow(im);ax.add_patch(Rectangle((left*w/size[0],top*h/size[1]),224*w/size[0],224*h/size[1],fill=False,color='red',lw=2));ax.set_title(r['case_id']+' full source + fixed crop')
  axs[j,1].imshow(crop);axs[j,1].set_title('Source actually represented by cached features')
  axs[j,2].imshow(crop);ids=supports[r['source_group']]['subject'];mask=np.zeros(256);mask[ids]=1;mask=np.ma.masked_where(mask.reshape(16,16)==0,mask.reshape(16,16));axs[j,2].imshow(mask,extent=(-.5,223.5,223.5,-.5),interpolation='nearest',cmap='autumn',vmin=0,vmax=1,alpha=.45);axs[j,2].set_title('Existing selected source patches (no new mask)')
  for ax in axs[j]:ax.axis('off')
 fig.suptitle('Source-only geometry review; no pair scores, winners or expert labels');fig.tight_layout();fig.savefig(run/'qa'/f'source_support_{page+1}.png',dpi=130);plt.close(fig)
print('Rendered source-only support sheets; no preference scores or labels read.')
