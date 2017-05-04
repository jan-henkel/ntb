from itertools import product
import numpy as np
from .graph import ComputationGraph
from random import randrange
from copy import deepcopy

def num_grad(f,x,dx=2e-5):
    dfdx = np.zeros_like(x)
    for idx in product(*(range(n) for n in x.shape)):
        x[idx]+=dx/2
        df = f(x)
        x[idx]-=dx
        df -= f(x)
        x[idx]+=dx/2
        dfdx[idx] = df/dx
    return dfdx

def _rel_err(a,b,eps):
    return np.abs(a-b)/(np.abs(a)+np.abs(b)+eps)

def num_grad_sparse_error(f,x,analytic_grad,num_samples=10,dx_scale=1e-3,dx=2e-5,eps=1e-9):
    dfdx = np.zeros_like(x)
    dx = min(max(abs(x).mean(),eps)*dx_scale,dx)
    rel_error = []
    abs_error = []
    for i in range(num_samples):
        idx = tuple([randrange(m) for m in x.shape])
        #print(node,idx)
        f0 = f(x)
        tmp = x[idx]
        x[idx]=x[idx]+dx
        f1 = f(x)
        #print(x,f1)
        x[idx]=x[idx]-2*dx
        f2 = f(x)
        #print(x,f2)
        x[idx]=tmp
        grad_ana = analytic_grad[idx]
        grad_num_10 = (f1-f0)/dx
        grad_num_02 = (f0-f2)/dx
        grad_num_12 = (f1-f2)/(2*dx)
        rel_err_10 = _rel_err(grad_num_10,grad_ana,eps)
        rel_err_02 = _rel_err(grad_num_02,grad_ana,eps)
        rel_err_12 = _rel_err(grad_num_12,grad_ana,eps)
        abs_err_10 = np.abs(grad_num_10-grad_ana)
        abs_err_02 = np.abs(grad_num_02-grad_ana)
        abs_err_12 = np.abs(grad_num_12-grad_ana)
        #print(grad_ana,grad_num)
        rel_error.append(min([rel_err_10,rel_err_02,rel_err_12]))
        abs_error.append(min([abs_err_10,abs_err_02,abs_err_12]))
    #print(rel_error)
    return np.array(rel_error).mean(),np.array(abs_error).mean(),dx

def test_num_grads(loss_node,assign_dict,dx_scale=1e-3,dx=2e-5,num_samples=10,exclude=[]):
    graph = loss_node.graph
    nodes = [n for n in graph.bp_subtree[loss_node] if n.shape!=() and n not in exclude]
    #print(nodes)
    graph._flush()
    graph._assign(assign_dict)
    graph._diff(loss_node)
    grads = graph.d[loss_node].copy()
    evals = graph.fp_eval.copy()
    def _f(node):
        def f(x):
            assign_dict[node] = x
            result, = graph.run([loss_node],assign_dict=assign_dict)
            assign_dict.pop(node)
            return result
        x = evals[node].copy().astype(float)
        return f,x
    num_grad_errors = {n:num_grad_sparse_error(*_f(n),grads[n],dx_scale=dx_scale,dx=dx,num_samples=num_samples) for n in nodes}
    return evals,grads,num_grad_errors
