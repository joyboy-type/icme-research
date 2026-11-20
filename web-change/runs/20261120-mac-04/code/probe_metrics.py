import numpy as np
def class_mask(records,label):
    return np.asarray([r['label']==label and bool(r['positive']) and r['condition']=='clean' for r in records],dtype=bool)
