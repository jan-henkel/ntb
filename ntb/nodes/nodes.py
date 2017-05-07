import numpy as np
import copy
from itertools import zip_longest
from ..aux import shape_matches
from ..graph import ComputationGraph
from ..initializers import const_init,xavier_init

class default_graph:
    def __init__(self,graph):
        self.graph = graph
        
    def __enter__(self):
        default_graph.graph = self.graph

    def __exit__(self,type,value,traceback):
        delattr(default_graph,'graph')

class Node:

    """
    Classes derived from Node represent nodes in a computation graph. They implement a simple interface for forward and backpropagation.
    
    The __init__ procedure takes as arguments the input nodes, the computation graph (which is ignored if the tuple of input nodes is not empty and required otherwise),
    an iterable d_nodes of parameter nodes to be included for automatic differentiation. Furthermore there are arbitrary keyword arguments,
    which are passed along to self.init(**kwargs), a procedure meant to be customized in derived classes.

    self.forw_eval takes as arguments the inputs needed for the calculation and returns an output tensor as well as a cache of intermediate results which are helpful (or necessary) for the backward pass.
    self.back_eval takes as arguments the gradient Dout (e.g. of a loss function) with respect to the node's output, the aforementioned cache and a list bp_mask of booleans, one for each input.
    It returns gradients wrt the inputs in the same order in which they are passed to forw_eval. bp_mask lets back_eval know which of these gradients are required (corresponding entry of bp_mask is True),
    the rest of them may be set to any value including None.
    self.infer_shape calculates the shape of the output tensor given the input tensor shapes
    """

    def __init__(self,*input_nodes,graph:ComputationGraph = None,**kwargs):
        if len(input_nodes)>0:
            graph = input_nodes[0].graph
        else:
            graph = getattr(default_graph,'graph',None)
        assert graph is not None, "No graph specified"
        self.shape = kwargs.get('shape')
        if self.shape is not None:
            self.shape = tuple(self.shape)
        self.init(**kwargs)
        graph.add_node(self,list(input_nodes))

    def init(self,**kwargs):
        pass
    
    def forw_eval(self,*inputs):
        out=None
        cache=None
        return out,cache
    
    def back_eval(self,Dout,cache=None,bp_mask=[True]*0):
        Dinputs = tuple()
        return Dinputs

    def infer_shape(self,*input_shapes):
        if self.shape is not None:
            return self.shape
        else:
            return default_result_shape(*input_shapes)

    #We define a number of convenient operators and functions to get new nodes from existing ones.

    def __add__(self,other):
        try:
            val = np.float(other)
            return Function1d(self,f=lambda x:x+val,df=lambda x:1.)
        except:
            pass
        return Add(self,other)

    def __sub__(self,other):
        try:
            val = np.float(other)
            return Function1d(self,f=lambda x:x-val,df=lambda x:1.)
        except:
            pass
        return Sub(self,other)

    def __mul__(self,other):
        try:
            val = np.float(other)
            return Function1d(self,f=lambda x:x*val,df=lambda x:val)
        except:
            pass
        return Mult(self,other)
    
    def __truediv__(self,other):
        try:
            val = np.float(other)
            return Function1d(self,f=lambda x:x/val,df=lambda x:1./val)
        except:
            pass
        return Div(self,other)

    def __pow__(self,other):
        return Function1d(self,f=lambda x:x**other,df=lambda x:other*x**(other-1))

    def __radd__(self,other):
        return self.__add__(other)
    
    def __rmul__(self,other):
        return self.__mul__(other)

    def __rsub__(self,other):
        try:
            val = np.float(other)
            return Function1d(self,f=lambda x:val-x,df=lambda x:-1.)
        except:
            pass
        return Sub(other,self)

    def __rdiv__(self,other):
        try:
            val = np.float(other)
            return Function1d(self,f=lambda x:val/x,df=lambda x:-val/x**2)
        except:
            pass
        return Div(other,self)

    def __neg__(self):
        return Function1d(self,f=lambda x:-x,df=lambda x:-1.)

    def __getitem__(self,index):
        return Subscript(self,index=index)
    
    def dot(self,other):
        return Dot(self,other)

    def sum(self,axis=None):
        return Sum(self,axis=axis)

    def mean(self,axis=None):
        return Mean(self,axis=axis)

def default_result_shape(*input_shapes):
    m = max(len(s) for s in input_shapes)
    normalized = {(1,)*(m-len(s))+s for s in input_shapes}
    res = ()
    for i in range(m):
        si = {s[i] for s in normalized}
        li = list(si.difference({1,-1}))
        if(len(li)>1):
            return None
        elif(len(li)==1):
            res += (li[0],)
        else:
            res += (min(si),)
    return res
    
class Variable(Node):

    def init(self,initializer=None,value=None,learnable=True,**kwargs):
        if initializer is None:
            if value is None:
                if self.shape is None:
                    self.shape = ()
                initializer = xavier_init(self.shape)
            else:
                initializer = const_init(value)
        def _reset():
            self.value = initializer()
        self.reset = _reset
        self.reset()
        
        if self.shape is None:
            self.shape = np.shape(self.value)

        assert shape_matches(self.shape,np.shape(self.value)), "Initial value shape does not match specified shape"

        self.learnable = learnable
        self.mutable = True
        
    def forw_eval(self):
        return self.value,None

    def back_eval(self,Dout,cache=None,bp_mask=None):
        return ()

    def save_attributes(self):
        return self.value.copy()

    def load_attributes(self,v):
        self.value = v.copy()

class Placeholder(Node):

    def init(self,**kwargs):
        if self.shape is None:
            self.shape = ()

    def forw_eval(self):
        raise Exception('No value assigned to Placeholder node')

    def back_eval(self,Dout,cache=None,bp_mask=None):
        raise Exception("Backpropagating from a Placeholder Node")

def sum_to_match(x,shape):
    if shape == ():
        return np.sum(x)
    elif shape == x.shape:
        return x
    else:
        diff = len(x.shape)-len(shape)
        shape = (0,)*diff+shape
        return np.sum(x,axis=[i for i in range(len(x.shape)) if shape[i]<x.shape[i]])

def fix_ds(dx,dy,sx,sy,bp_mask):
    if sx==sy:
        return dx,dy
    else:
        if bp_mask[0]: dx=sum_to_match(dx,sx)
        if bp_mask[1]: dy=sum_to_match(dy,sy)
        return dx,dy
    
class Add(Node):

    def forw_eval(self,x,y):
        return x+y,(np.shape(x),np.shape(y))
    
    def back_eval(self,Dout,cache,bp_mask=None):
        sx,sy = cache
        dx,dy = Dout,Dout
        return fix_ds(dx,dy,sx,sy,bp_mask)
    
class Sub(Node):

    def forw_eval(self,x,y):
        return x-y,(np.shape(x),np.shape(y))
    
    def back_eval(self,Dout,cache,bp_mask=[True]*2):
        sx,sy = cache
        dx,dy=Dout,-Dout
        return fix_ds(dx,dy,sx,sy,bp_mask)

class Mult(Node):

    def forw_eval(self,x,y):
        return x*y,(x,y)

    def back_eval(self,Dout,cache,bp_mask=[True]*2):
        x,y = cache
        sx,sy = np.shape(x),np.shape(y)
        dx,dy = bp_mask
        if dx: dx=Dout*y
        if dy: dy=Dout*x
        return fix_ds(dx,dy,sx,sy,bp_mask)

class Div(Node):

    def forw_eval(self,x,y):
        yinv = 1./y
        return x*yinv,(x,yinv)

    def back_eval(self,Dout,cache,bp_mask=[True]*2):
        x,yinv = cache
        sx,sy = np.shape(x),np.shape(yinv)
        dx,dy = bp_mask
        if dx: dx=Dout*yinv
        if dy: dy=-Dout*x*(yinv**2)
        return fix_ds(dx,dy,sx,sy,bp_mask)
    
class Dot(Node):

    def infer_shape(self,sx,sy):
        assert sx[1] in (-1,sy[0]) and len(sx)==len(sy)==2
        return (sx[0],sy[1])

    def forw_eval(self,x,y):
        return x.dot(y),(x.t,y.t)
    
    def back_eval(self,Dout,cache,bp_mask=[True]*2):
        xt,yt = cache
        dx,dy = bp_mask
        if dx: dx=Dout.dot(yt)
        if dy: dy=xt.dot(Dout)
        return dx,dy

class Affine(Node):

    def infer_shape(self,sx,sy,sb):
        assert len(sx)==len(sy)==2 and len(sb)==1
        assert (sx[1] in (-1,sy[0])) and (sb[0] in (-1,sy[1]))
        return (sx[0],sy[1])
    
    def forw_eval(self,x,y,b):
        return x.dot(y)+b,(x.T,y.T)

    def back_eval(self,Dout,cache,bp_mask=[True]*3):
        xt,yt = cache
        dx,dy,db = bp_mask
        if dx: dx=Dout.dot(yt)
        if dy: dy=xt.dot(Dout)
        if db: db=Dout.sum(axis=0)
        return dx,dy,db
    
class Function1d(Node):

    def init(self,**kwargs):
        self.f,self.df = kwargs['f'], kwargs['df']

    def forw_eval(self,x):
        return self.f(x),x

    def back_eval(self,Dout,cache,bp_mask=[True]):
        dx, = bp_mask
        if dx:
            x = cache
            dx = Dout*self.df(x)
        return dx,

class Reshape(Node):

    def infer_shape(self,sx):
        p1 = np.prod(sx)
        p2 = np.prod(self.shape)
        if -1 in self.shape:
            assert p1 % p2 == 0
        else:
            assert p1 == p2
        return self.shape

    def forw_eval(self,x):
        return x.reshape(self.shape),x.shape

    def back_eval(self,Dout,cache,bp_mask=[True]):
        dx, = bp_mask
        if dx:
            shape_in = cache
            dx = Dout.reshape(shape_in)
        return dx,

class BatchFlatten(Node):

    def infer_shape(self,sx):
        if -1 not in sx[1:]:
            return (sx[0],np.prod(sx[1:]))
        else:
            return (sx[0],-1)

    def forw_eval(self,x):
        return x.reshape([x.shape[0],-1]),x.shape

    def back_eval(self,Dout,cache,bp_mask=[True]):
        dx, = bp_mask
        if dx:
            shape_in = cache
            dx = Dout.reshape(shape_in)
        return dx,

class Sum(Node):

    def init(self,axis=None):
        self.axis = axis

    def infer_shape(self,sx):
        if self.axis is None:
            return ()
        else:
            return sx[:self.axis]+sx[self.axis+1:]

    def forw_eval(self,x):
        return np.sum(x,axis=self.axis),x.shape

    def back_eval(self,Dout,cache,bp_mask=[True]):
        dx, = bp_mask
        if dx:
            shape_in = cache
            if self.axis is not None:
                dx = (np.ones(shape_in).swapaxes(0,self.axis)*Dout).swapaxes(0,self.axis)
            else:
                dx = np.ones(shape_in)*Dout
        return dx,

class Mean(Node):

    def init(self,axis=None):
        self.axis = axis

    def infer_shape(self,sx):
        if self.axis is None:
            return ()
        else:
            return sx[:self.axis]+sx[self.axis+1:]
    
    def forw_eval(self,x):
        return np.mean(x,axis=self.axis),x.shape

    def back_eval(self,Dout,cache,bp_mask=[True]):
        dx, = bp_mask
        if dx:
            shape_in = cache
            if self.axis is not None:
                dx = (np.ones(shape_in).swapaxes(0,self.axis)*Dout/shape_in[self.axis]).swapaxes(0,self.axis)
            else:
                dx = np.ones(shape_in)*Dout/np.prod(shape_in)
        return dx,

class Argmax(Node):

    def init(self,axis):
        self.axis=axis

    def infer_shape(self,sx):
        return sx[:self.axis]+sx[self.axis+1:]

    def forw_eval(self,x):
        return np.argmax(x,axis=self.axis),None

    def back_eval(self,Dout,cache,mask):
        return 0,

class Sample(Node):

    def infer_shape(self,sp):
        assert len(sp) in {2,3}
        return sp[:-1]

    def forw_eval(self,p):
        p_reshaped = p.reshape(-1,p.shape[-1])
        return np.array([np.random.choice(p.shape[-1],p=r) for r in p_reshaped]).reshape(p.shape[:-1]), None

    def back_eval(self,Dout,cache,mask):
        return 0,

class Equals(Node):

    def forw_eval(self,x,y):
        return x==y,None

    def back_eval(self,Dout,cache,mask):
        return 0,0,

class Subscript(Node):

    def init(self,index): #,reduce_shape=False):
        self.index = index
        #self.reduce_shape = reduce_shape
    
    def infer_shape(self,sx):
        if type(self.index) is tuple:
            assert len(sx)>=len(self.index),"Invalid slice or index"
            #self.shape_mask = np.array([0 if type(i) is int else 1 for d,i in zip_longest(sx,self.index)])
            res = ()
            for d,i in zip_longest(sx,self.index,fillvalue=slice(None)):
                if type(i) is slice:
                    if d==-1:
                        res+=(-1,)
                    else:
                        slice_len = i.indices(d)[1]-i.indices(d)[0]
                        assert slice_len>0,"Invalid slice"
                        res+=(slice_len,)
                else:
                    assert type(i) is int, "Invalid index"
            return res
        else:
            assert len(sx)>1
            #self.shape_mask = np.array([0]+[1]*(len(sx)-1))
            return sx[1:]

    def forw_eval(self,x):
        return x[self.index],x.shape

    def back_eval(self,Dout,cache,bp_mask=[True]):
        shape_x = cache
        """
        if self.reduce_shape:
            reduced_shape = np.array(shape_x)*self.shape_mask+(1-self.shape_mask) #equals 1 in dimensions where an integer index was passed, same as input shape elsewhere
            dx = np.zeros(reduced_shape)
        else:
            dx = np.zeros(shape_x)
        """
        dx = np.zeros(shape_x)
        dx[self.index] = Dout
        return dx,
