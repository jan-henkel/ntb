import numpy as np
from itertools import compress
from .misc import shape_matches
        
class ComputationGraph:
    """
    Usage:

    ComputationGraph implements forward and backpropagation in computation graphs. 
    
    Nodes are added with add_node(node,inputs,d_inputs) whereby inputs is a list of input nodes and
    d_inputs is a list of scalar nodes whose derivatives are required to evaluate the node
    
    run(nodes,assign_dict) takes a list of nodes and a dict of assignments {variable_node:numpy_array}
    and returns the requested computation results, given the placeholder assignments.

    Internals:

    A ComputationGraph object has the following members:
    - a set self.nodes and a list self.mutable_nodes of nodes with the 'mutable' attribute
    - for each node n its ingoing and outgoing connections/edges are stored in self.inputs[n] and self.outputs[n] respectively
    - for each node n a list of scalar nodes whose derivatives are required to evaluate n is stored in self.d_inputs[n]
    - for each node n visited during a forward pass, we record the following data:
        - self.fp_eval[n], the actual result of the computation at n
        - self.fp_cache[n], intermediate results that are helpful for the backward pass at n
    - for each differentiable scalar node o, self.bp_subtree[o] is a set of nodes to be visited in a backward pass
      (this is to ensure that gradients aren't needlessly propagated to input leafnodes that don't correspond to learnable parameters)
    - during backprop starting at an output node o:
        - a set self.bp_done[o] stores all nodes that have been fully processed
        - for each node n visited, self.d[o][n] holds the derivative of o with respect to n
    """

    def __init__(self):
        self.nodes = set()
        self.mutable_nodes = list()
        self.inputs = dict()
        self.d_inputs = dict()
        self.outputs = dict()
        
        self.fp_eval = dict()
        self.fp_cache = dict()
        
        self.d = dict()
        
        self.bp_done = dict()
        self.bp_subtree = dict()

    def _flush(self):
        #clear all cached results (necessary when new input is provided)
        self.fp_eval.clear()
        self.fp_cache.clear()
        self.d.clear()
        self.bp_done.clear()

    def _eval(self,node):
        if node not in self.nodes:
            raise Exception("Node not present in this graph.")

        #only perform calculation if result isn't already available
        try:
            return self.fp_eval[node]
        except:
            pass
        
        #(recursively) evaluate inputs and feed the results into the node-specific forw_eval, store result and cache of intermediate results
        x, cache = node.forw_eval(*(self._eval(n) for n in self.inputs[node]),*(self._diff(n) for n in self.d_inputs[node]))
        self.fp_eval[node]=x
        self.fp_cache[node]=cache
        
        return x

    def _diff(self,outnode):
        assert outnode in self.nodes, "Node not present in this graph."
        assert outnode.shape == (), "Node is not a scalar."
        assert outnode in self.bp_subtree, "Node is not differentiable"

        #only perform calculation if result isn't already available
        try:
            return self.d[outnode]
        except:
            pass

        #reset derivatives of outnode, processed nodes in subtree
        self.d[outnode]=dict()
        self.bp_done[outnode]=set()

        #forward pass
        self._eval(outnode)

        #backpropagate to find derivatives
        self.d[outnode][outnode] = 1.
        self._backprop(outnode,outnode)
        return self.d[outnode]

    #helper functions to recursively collect nodes along ingoing or outgoing edges and store them in a set s
    
    def _input_subtree_nodes(self,node,s):
        #recursively collect input nodes
        if node not in s:
            s.add(node)
            for n in self.inputs[node]:
                self._input_subtree_nodes(n,s)

    def _output_subtree_nodes(self,node,s):
        #recursively collect output nodes
        if node not in s:
            s.add(node)
            for n in self.outputs[node]:
                self._output_subtree_nodes(n,s)
                
    def _backprop(self,node,outnode):
        #if node value resulted from direct assignment (as opposed to forward pass from inputs), return
        if node not in self.fp_cache.keys():
            self.bp_done[outnode].add(node)
            return

        #if node was already processed return
        if node in self.bp_done[outnode]:
            return

        #check if backpropagation along all outgoing edges of node in the relevant subtree is done, if not, return
        if any(n not in self.bp_done[outnode] and n in self.bp_subtree[outnode] for n in self.outputs[node]):
            return

        #obtain backpropagated gradients from outgoing gradient d[outnode][node]
        #only propagate along inbound edges within the relevant subtree
        mask=[n in self.bp_subtree[outnode] for n in self.inputs[node]]
        ds = node.back_eval(self.d[outnode][node],self.fp_cache[node],mask)
        relevant_ds = tuple(compress(ds,mask))
        relevant_inputs = tuple(compress(self.inputs[node],mask))
        for n,d in zip(relevant_inputs,relevant_ds):
            if n in self.d[outnode].keys():
                self.d[outnode][n]=self.d[outnode][n]+d
            else:
                self.d[outnode][n]=d
        
        #done processing this node
        self.bp_done[outnode].add(node)

        #tentatively call backprop on inputs of node
        #(does nothing for nodes whose outbound edges haven't all been processed)
        for n in relevant_inputs:
            self._backprop(n,outnode)
    
    def _assign(self,assign_dict):
        for n,v in assign_dict.items():
            assert shape_matches(n.shape,np.shape(assign_dict[n]))
            self.fp_eval[n]=v

    def add_node(self,node,inputs=list(),d_inputs=list()):
        assert node not in self.nodes, "Node already present in this graph."
        assert all(n in self.nodes for n in inputs) and all(n in self.nodes for n in d_inputs),"Not all input nodes are present in this graph."
        assert not any(getattr(n,'disruptive',False) for n in inputs) and not any(getattr(n,'disruptive',False) for n in d_inputs), "Nodes with the 'disruptive' flag cannot serve as inputs to other nodes."
        
        #update dicts to accomodate the new node
        self.inputs[node]=inputs
        self.d_inputs[node]=d_inputs
        self.outputs[node]=list()
        if getattr(node,'shape',None) is None:
            node.shape = node.infer_shape(*(n.shape for n in inputs))
        self.nodes.add(node)
        if getattr(node,'mutable',False):
            self.mutable_nodes.append(node)
        node.graph = self
        for n in inputs:
            self.outputs[n].append(node)

        #for scalar nodes, build a subtree for automatic differentiation
        if node.shape==():
            self.set_d_nodes(node)

    #for a scalar output node, set a list of nodes to be considered for
    #automatic differentiation (defaults to learnable parameters)
    #prune the subtree accordingly, to avoid superfluous calculations
    def set_d_nodes(self,node,d_nodes=None):
        #recursively collect input nodes
        self.bp_subtree[node]=set()
        self._input_subtree_nodes(node,self.bp_subtree[node])

        if d_nodes is None:
            d_nodes = [n for n in self.bp_subtree[node] if getattr(n,'learnable',False)]
    
        out_subtreenodes = set()
        for n in d_nodes:
            self._output_subtree_nodes(n,out_subtreenodes)
        self.bp_subtree[node].intersection_update(out_subtreenodes)
        if not all(getattr(n,'differentiable',True) for n in self.bp_subtree[node]):
            self.bp_subtree.pop(node)

    #compute results for a list of nodes, given a dict of assignments
    def run(self,nodes,assign_dict):
        self._flush()
        self._assign(assign_dict)
        def _run(node):
            r = self._eval(node)
            #if evaluating the node changes something about the graph, reset everything
            #(only current example is an optimizer)
            if getattr(node,'disruptive',False):
                self._flush()
                self._assign(assign_dict)
            return r
        return tuple(_run(n) for n in nodes)

    #save all the moving parts of the graph in a list
    def save(self):
        return [n.save_attributes() for n in self.mutable_nodes]

    #load from a list
    def load(self,l):
        for n,a in zip(self.mutable_nodes,l):
            n.load_attributes(a)

    #reset all mutable parts
    def reset(self):
        for n in self.mutable_nodes:
            n.reset()
