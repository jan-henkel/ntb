import matplotlib.pyplot as pp
import numpy as np
from itertools import product

def render_cnn_filters(w,**kwargs):
    pp.rcParams['image.interpolation'] = kwargs.get('interpolation','nearest')
    pp.rcParams['figure.figsize'] = kwargs.get('figsize',(9.0, 9.0))
    pp.rcParams['image.cmap'] = kwargs.get('cmap','gray')
    
    F,C,HH,WW=w.shape
    filters = w.reshape(F*C,HH,WW)
    for i,f,x,y in zip(range(F*C),filters,*zip(*product(range(F),range(C)))):
        pp.subplot(C,F,i+1)
        pp.imshow(f)
        pp.axis('off')
    pp.show()
