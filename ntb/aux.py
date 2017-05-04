import numpy as np

def shape_matches(s1,s2):
    return s1 is None or (len(s1)==len(s2) and all(a==b or a==-1 for a,b in zip(s1,s2)))

def sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)
