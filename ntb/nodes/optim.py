from .nodes import Node,Variable
from ..graph import ComputationGraph
from ..initializers import const_init
import numpy as np
from copy import deepcopy
"""
An optimization node when evaluated modifies the values of learnable parameters (Variable nodes with self.learnable=True)
so as to minimize the loss.
"""

class Optim(Node):
    
    def __init__(self,loss_node:Node,**kwargs):
        graph = loss_node.graph
        assert graph is not None, "No graph specified"

        #if necessary, convert learning rate to a node in the graph
        lr=kwargs.pop('lr',1e-4)
        try:
            lr = np.float64(lr)
            lr = Variable(initializer=const_init(lr),learnable=False,graph=graph)
        except:
            pass

        #set up update rule
        self.update_rule = kwargs.pop('update_rule','sgd')
        if not hasattr(self,self.update_rule):
            raise Exception('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule =getattr(Optim,self.update_rule)

        #set up default optimization config
        self.config = kwargs.pop('config',dict())

        #find all nodes that need to be optimized
        try:
            #self.nodes = [n for n in graph.bp_subtree[loss_node] if getattr(n,'learnable',False)]
            self.nodes = [n for n in graph.mutable_nodes if n in graph.bp_subtree[loss_node] and getattr(n,'learnable',False)]
        except:
            raise Exception("Not a valid loss node")
        
        self.reset()

        #set 'disruptive' flag, because changes are made to other nodes in the graph when this node is evaluated
        self.disruptive = True
        #set 'mutable' flag, because this node has moving parts that could be saved and restored
        self.mutable = True

        #add self to graph
        graph.add_node(self,[lr],[loss_node])

    def reset(self):
        self.configs = [{a:b for a,b in self.config.items()} for n in self.nodes]
        #for n in self.nodes:
        #    self.configs[n] = {a:b for a,b in self.config.items()}

    def save_attributes(self):
        return deepcopy(self.configs)

    def load_attributes(self,configs):
        self.configs = deepcopy(configs)

    def infer_shape(self,slr):
        assert slr == ()
        return ()
        
    def forw_eval(self,lr,d_loss):
        for n,c in zip(self.nodes,self.configs):
            n.value = self.update_rule(n.value,d_loss.get(n,0),lr,c)
        return True,None

    def back_eval(self,Dout,cache=None,bp_mask=None):
        raise Exception("Backpropagating from an optimization node")

    def sgd(w,dw,lr,config):
        m=config.setdefault('momentum',.9)
        v=config.setdefault('velocity',np.zeros_like(w))
        v=v*m-lr*dw
        config['velocity'] = v
        return w+v

    def rmsprop(w,dw,lr,config):
        dec=config.setdefault('decay_rate',.99)
        eps=config.setdefault('epsilon',1e-8)
        cache=config.setdefault('cache',np.zeros_like(w))
        cache = dec*cache+(1.-dec)*(dw**2)
        next_w = w-lr*dw/(np.sqrt(cache)+eps)
        config['cache'] = cache
        return next_w

    def adam(w,dw,lr,config):
        beta1=config.setdefault('beta1',.9)
        beta2=config.setdefault('beta2',.999)
        eps=config.setdefault('epsilon',1e-8)
        config.setdefault('m',np.zeros_like(w))
        config.setdefault('v',np.zeros_like(w))
        config.setdefault('t',0)
        c=config
        c['t']=c['t']+1
        c['m']=beta1*c['m']+(1-beta1)*dw
        m0=c['m']/(1-beta1**c['t'])
        c['v']=beta2*c['v']+(1-beta2)*(dw**2)
        v0=c['v']/(1-beta2**c['t'])
        next_w = w-lr*m0/(np.sqrt(v0)+eps)
        return next_w
