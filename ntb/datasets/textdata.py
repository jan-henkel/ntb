import numpy as np
import os
from .data_utils import split_array

def load(filename,seq_length=50,path='./ntb/datasets/textdata/',val_frac=.05,stride=None):

    if stride is None:
        stride = seq_length
    
    with open(os.path.join(path, filename),'r') as f:
        txt = f.read()
    
    charmap = sorted(list(set(txt)))
    inv_charmap = {c:i for i,c in enumerate(charmap)}

    def decode(a):
        return "".join(list(map(lambda i:charmap[i],a)))

    def encode(s):
        return np.array(list(map(lambda c:inv_charmap[c],s)))


    segments = [encode(txt[i:i+seq_length+1]) for i in range(0,len(txt)-seq_length-1,stride)]
    
    X = np.array([s[:-1] for s in segments])
    y = np.array([s[1:] for s in segments])
    X_train,X_val = split_array(X,[(1-val_frac),val_frac])
    y_train,y_val = split_array(y,[(1-val_frac),val_frac])
    if stride<seq_length:
        X_val=X_val[((seq_length-1)//stride)-1:]
        y_val=y_val[((seq_length-1)//stride)-1:]

    return {'X_train':X_train,
            'y_train':y_train,
            'X_val':X_val,
            'y_val':y_val,
            'charmap':charmap,'inv_charmap':inv_charmap},encode,decode
