from .nodes import Node
import numpy as np

class CrossEntropy(Node):

    def infer_shape(self,sp,sy):
        N,C = sp
        assert sy == (N,)
        return ()

    def forw_eval(self,p,y):
        N = p.shape[0]
        p1 = p[np.arange(N),y]
        loss = np.mean(-np.log(p1))
        return loss,(p,p1,y,N)

    def back_eval(self,Dout,cache,bp_mask=[True]*2):
        p,p1,y,N = cache
        dp,dy = bp_mask
        if dp:
            dp = np.zeros_like(p)
            dp[np.arange(N),y] = -1./p1*Dout/N
        if dy: dy = 0
        return dp,dy

class CrossEntropyLogits(Node):

    def infer_shape(self,sx,sy):
        N,C = sx
        assert sy == (N,)
        return ()

    def forw_eval(self,x,y):
        #eps = 1e-13
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))#+eps
        probs /= np.sum(probs, axis=1, keepdims=True)
        N = x.shape[0]
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        return loss,(probs,N,y)

    def back_eval(self,Dout,cache,bp_mask=[True,False]):
        probs,N,y=cache
        dx,dy = bp_mask
        if dx:
            dx = probs.copy()
            dx[np.arange(N), y] -= 1
            dx /= N
            dx *= Dout
        if dy: dy = 0
        return dx,dy

class ClassAccuracy(Node):

    def infer_shape(self,spred_y,sy):
        N, = spred_y
        assert sy == (N,)
        return ()

    def forw_eval(self,pred_y,y):
        return np.mean(pred_y==y),None

    def back_eval(self,Dout,cache,bp_mask):
        raise Exception('Accuracy node not differentiable')
