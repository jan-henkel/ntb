from ..aux import sigmoid
from .nodes import Node,Function1d
import numpy as np

class Relu(Node):

    def forw_eval(self,x):
        return np.maximum(x,0.),x

    def back_eval(self,Dout,cache,bp_mask=[True]):
        dx,=bp_mask
        if dx:
            x = cache
            dx = Dout*(x>0.)
        return dx,

class Sigmoid(Node):

    def forw_eval(self,x):
        s = sigmoid(x)
        return s,s

    def back_eval(self,Dout,cache,bp_mask=[True]):
        dx, = bp_mask
        if dx:
            s = cache
            dx = Dout*s*(1-s)
        return dx,

class Tanh(Node):

    def forw_eval(self,x):
        t = np.tanh(x)
        return t,t

    def back_eval(self,Dout,cache,bp_mask=[True]):
        dx, = bp_mask
        if dx:
            t = cache
            dx = Dout*(1-t**2)
        return dx,

class Maxout(Node):

    def infer_shape(self,sx):
        return sx[:-1]

    def forw_eval(self,x):
        m=np.max(x,axis=-1)
        return m,(m,x)

    def back_eval(self,Dout,cache,bp_mask=[True]):
        dx, = bp_mask
        if dx:
            m,x = cache
            dx = Dout*(x.swapaxes(-1,0) == m).swapaxes(-1,0)
        return dx,


class Softmax(Node):

    def forw_eval(self,x):
        #side note: x-=np.max(x) would actually alter the numpy tensor x passed to the function
        x=x-np.max(x,axis=-1,keepdims=True)
        r=np.exp(x)
        s=r/np.sum(r,axis=-1,keepdims=True)
        return s,s

    def back_eval(self,Dout,cache,bp_mask=[True]):
        dx,=bp_mask
        if dx:
            s=cache
            dx=(Dout-np.sum(Dout*s,axis=-1,keepdims=True))*s
        return dx,
