"""Fixed local residual descriptors; inputs are aligned RGB screenshots only."""
import cv2
import numpy as np

def describe(before,after,margin=4):
    if before.shape!=after.shape or min(before.shape[:2])<=2*margin:raise ValueError('Invalid paired image shapes')
    area=(slice(margin,-margin),slice(margin,-margin)) if margin else (slice(None),slice(None))
    raw=np.abs(before.astype(np.float32)-after.astype(np.float32)).max(axis=2)[area]
    a,b=[cv2.GaussianBlur(x,(3,3),.6) for x in [before,after]]
    blur=np.abs(a.astype(np.float32)-b.astype(np.float32)).max(axis=2)[area]
    values=[];names=[];scalars={}
    def add(name,x):names.append(name);values.append(float(x))
    for prefix,d in [('raw',raw),('blur',blur)]:
        add(prefix+'_mean',d.mean()/255);add(prefix+'_std',d.std()/255);add(prefix+'_max',d.max()/255)
        for q in [.9,.99,.999]:add(prefix+'_q'+str(q),np.quantile(d,q)/255)
        for threshold in [4,8,16,32,64]:
            mask=(d>threshold).astype(np.uint8);add(f'{prefix}_fraction{threshold}',mask.mean())
            n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
            sizes=stats[1:,cv2.CC_STAT_AREA] if n>1 else np.zeros(0)
            largest=float(sizes.max()) if len(sizes) else 0.
            add(f'{prefix}_component_count{threshold}',len(sizes)/1000)
            add(f'{prefix}_largest{threshold}',largest/mask.size)
            add(f'{prefix}_concentration{threshold}',largest/max(1,float(mask.sum())))
            if len(sizes):
                best=stats[1:][int(np.argmax(sizes))]
                add(f'{prefix}_largest_width{threshold}',best[cv2.CC_STAT_WIDTH]/mask.shape[1])
                add(f'{prefix}_largest_height{threshold}',best[cv2.CC_STAT_HEIGHT]/mask.shape[0])
            else:
                add(f'{prefix}_largest_width{threshold}',0);add(f'{prefix}_largest_height{threshold}',0)
            if prefix=='blur' and threshold==8:
                scalars['area_blur8']=float(mask.mean());scalars['largest_component8']=largest/mask.size
        for size in [16,32,64]:
            kernel=(min(size,d.shape[1]),min(size,d.shape[0]))
            local=cv2.boxFilter(d/255,cv2.CV_32F,kernel,normalize=True,borderType=cv2.BORDER_REFLECT)
            add(f'{prefix}_local{size}_max',local.max());add(f'{prefix}_local{size}_q99',np.quantile(local,.99))
            if prefix=='blur' and size==32:scalars['max_local32']=float(local.max())
    ga,gb=[cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in [before,after]]
    ea,eb=[cv2.Canny(x,40,100)[area]>0 for x in [ga,gb]]
    for label,plane in [('edge_new',eb&~ea),('edge_old',ea&~eb),('edge_shared',ea&eb)]:
        add(label+'_density',plane.mean())
    add('before_gray_std',ga[area].std()/255)
    return np.asarray(values,np.float64),names,scalars
