from .nodes import Node
from ..graph import ComputationGraph
import numpy as np

class Grad(Node):
    def __init__(self,y,x):
        graph : ComputationGraph = y.graph
        self.differentiable = False
        self.x = x
        graph.add_node(self,[x],[y])

    def infer_shape(self,sx):
        return sx

    def forw_eval(self,x,d_y):
        return d_y.get(self.x,np.zeros_like(x)),None

    def back_eval(self,Dout,cache,bp_mask):
        raise Exception("Backpropagating from a non-differentiable Grad node")
