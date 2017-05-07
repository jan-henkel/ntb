import numpy as np
from .nodes import Node,BatchFlatten,Variable,Affine,Conv2d,MaxPool,Batchnorm,get_pad,SpatialBatchnorm
from .initializers import xavier_init,const_init

"""
A number of auxiliary functions to construct neural network layers (with sensible defaults)
"""

def affine_layer(x,output_dim):
    g=x.graph
    if len(x.shape)>2:
        x = BatchFlatten(x)
    wshape = (x.shape[1],output_dim)
    w=Variable(initializer=xavier_init(wshape),graph=g)
    b=Variable(initializer=const_init(0.,(output_dim,)),graph=g)
    out = Affine(x,w,b)
    return out,w,b

def conv_layer(x,F,HH,WW,stride=1,padding='same'):
    assert padding in ('same','minimal')
    g=x.graph
    N,C,H,W = x.shape
    w=Variable(initializer=xavier_init((F,C,HH,WW)),graph=g)
    b=Variable(initializer=const_init(1e-1,(F,)),graph=g)
    pad = get_pad(H,W,HH,WW,stride,padding_type=padding)
    out = Conv2d(x,w,b,stride=stride,pad=pad)
    return out,w,b

def max_pool_layer(x,HH,WW,stride=None):
    N,C,H,W = x.shape
    if stride is None:
        stride = WW
    pad = get_pad(H,W,HH,WW,stride,padding_type='minimal')
    out = MaxPool(x,pool_size=(HH,WW),pad=pad,stride=stride)
    return out

def batchnorm_layer(x,train):
    g=x.graph
    gamma = Variable(initializer=const_init(1.,x.shape[1:]),graph=g)
    beta = Variable(initializer=const_init(0.,x.shape[1:]),graph=g)
    out = Batchnorm(x,gamma,beta,train)
    return out,gamma,beta

def spatial_batchnorm_layer(x,train):
    g=x.graph
    gamma = Variable(initializer=const_init(1.,(x.shape[1],)),graph=g)
    beta = Variable(initializer=const_init(0.,(x.shape[1],)),graph=g)
    out = SpatialBatchnorm(x,gamma,beta,train)
    return out,gamma,beta
