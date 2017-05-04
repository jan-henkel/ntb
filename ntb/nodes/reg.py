from .nodes import Node,default_result_shape,default_graph
import numpy as np

class Batchnorm(Node):

    def init(self,**kwargs):
        #learning rate for batch mean and variance
        self.mu = np.array(kwargs.get('mu',0.))
        self.var = np.array(kwargs.get('var',1.))
        self.eps = np.array(kwargs.get('eps',1e-5))
        self.lr = np.array(kwargs.get('learning_rate', .1))
        self.mutable = True
        self.defaults = self.save_attributes()

    def save_attributes(self):
        return tuple(v.copy() for v in (self.mu,self.var,self.eps,self.lr))

    def load_attributes(self,c):
        self.mu,self.var,self.eps,self.lr = (v.copy() for v in c)

    def reset(self):
        self.load_attributes(self.defaults)
        
    def infer_shape(self,sx,sg,sb,strain):
        assert sx[1:] == sb == sg
        return sx
    
    def forw_eval(self,x,g,b,train):
        if train:
            mubatch = x.mean(axis=0)
            xc = x-mubatch
            varbatch = np.mean(xc**2,axis=0)
            sigmabatch = np.sqrt(varbatch+self.eps)
            xn = xc/sigmabatch
            self.mu = self.mu*(1.-self.lr)+mubatch*self.lr
            self.var = self.var*(1.-self.lr)+varbatch*self.lr
            z = xn
            cache = (train,xc,sigmabatch,g)
        else:
            xc = x-self.mu
            sigma = np.sqrt(self.var+self.eps)
            xn = xc/sigma
            z = xn
            cache = (train,z,sigma,g)
        if g is not None:
            z = g*z
        if b is not None:
            z = z+b
        return z, cache

    def back_eval(self,Dout,cache,bp_mask=[True,True,True,False]):
        train = cache[0]
        dx,dg,db,dtrain = bp_mask

        #backpropagating dtrain might seem messy but shouldn't do any harm and allows us to use another node as input for train
        if dtrain: dtrain = 0
        
        if train:
            _,xc,sigmabatch,g = cache
            if dx: dx = g*((Dout-Dout.mean(axis=0))/sigmabatch-np.mean(Dout*xc,axis=0)*xc/sigmabatch**3)
            if dg: dg = np.sum(Dout*xc/sigmabatch,axis=0)
            if db: db = np.sum(Dout,axis=0)
        else:
            _,z,sigma,g = cache
            if dx: dx = g*Dout/sigma
            if dg: dg = np.sum(Dout*z,axis=0)
            if db: db = np.sum(Dout,axis=0)
        return dx,dg,db,dtrain

class TemporalBatchnorm(Batchnorm):

    def infer_shape(self,sx,sg,sb,strain):
        assert (sx[2],) == sb == sg
        return sx

    def forw_eval(self,x,g,b,train):
        N,T,D = x.shape
        out_reshaped, cache = Batchnorm.forw_eval(self,x.reshape(N*T,D),g,b,train)
        out = out_reshaped.reshape(N,T,D)
        return out,cache

    def back_eval(self,Dout,cache,bp_mask=[True,True,True,False]):
        N,T,D = Dout.shape
        dx,dg,db,dtrain = Batchnorm.back_eval(self,Dout.reshape(N*T,D),cache,bp_mask)
        if bp_mask[0]:
            dx = dx.reshape(N,T,D)
        return dx,dg,db,dtrain
    
class SpatialBatchnorm(Batchnorm):

    def infer_shape(self,sx,sg,sb,strain):
        assert (sx[1],) == sb == sg
        return sx
        
    def forw_eval(self,x,g,b,train):
        N,C,H,W = x.shape
        x_reordered = x.transpose(0,2,3,1).reshape(-1,C)
        out_reordered, cache = Batchnorm.forw_eval(self,x_reordered,g,b,train)
        out = out_reordered.reshape(N,H,W,C).transpose(0,3,1,2)
        return out,cache

    def back_eval(self,Dout,cache,bp_mask=[True,True,True,False]):
        N,C,H,W = Dout.shape
        Dout_reordered = Dout.transpose(0,2,3,1).reshape(-1,C)
        dx,dg,db,dtrain = Batchnorm.back_eval(self,Dout_reordered,cache,bp_mask)
        if bp_mask[0]:
            dx = dx.reshape(N,H,W,C).transpose(0,3,1,2)
        return dx,dg,db,dtrain
    
class Dropout(Node):

    def infer_shape(self,sx,sp):
        assert sp == ()
        return sx

    def forw_eval(self,x,p):
        d = np.random.binomial(n=1,p=p,size=x.shape)/p
        z = x*d
        cache = d
        return z,cache

    def back_eval(self,Dout,cache,mask=[True,False]):
        d = cache
        dx,dp = mask
        if dx: dx = Dout*d
        if dp: dp = 0
        return dx,dp

class L2(Node):

    def infer_shape(self,*input_shapes):
        return ()

    def forw_eval(self,*inputs):
        return .5*sum(np.sum(x**2) for x in inputs),inputs

    def back_eval(self,Dout,cache,mask):
        return tuple(Dout*x for x in cache)

class L1(Node):

    def infer_shape(self,*input_shapes):
        return ()

    def forw_eval(self,*inputs):
        return sum(np.sum(np.abs(x)) for x in inputs),inputs

    def back_eval(self,Dout,cache,mask):
        return tuple(Dout*np.sign(x) for x in cache)
    
def weight_decay(loss_node,decay_class=L2):
    graph = loss_node.graph
    input_nodes = [n for n in graph.mutable_nodes if n in graph.bp_subtree[loss_node] and getattr(n,'learnable',False) and len(n.shape)>1]
    return decay_class(*input_nodes)
