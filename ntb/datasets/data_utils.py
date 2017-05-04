import numpy as np

def split_array(array,fractions):
    splits = np.cumsum(fractions)
    assert splits[-1] == 1, "Fractions don't sum to 1"
    return np.split(array,(splits[:-1]*len(array)).astype(int))
