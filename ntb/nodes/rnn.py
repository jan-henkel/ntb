from .nodes import Node,Variable,default_graph
from ..initializers import xavier_init,normal_init,const_init
from ..aux import sigmoid
from itertools import compress
import numpy as np

class RnnCell(Node):

    def __iter__(self):
        return iter((self[0],self))

class Projection(Node):

    def init(self,n,i,shape):
        self.n,self.i,self.shape = n,i,shape

    def forw_eval(self,sx):
        return sx[self.i],None

    def back_eval(self,Dout,cache,bp_mask=[True]):
        r = np.array([0]*self.n,dtype=object)
        r[self.i] = Dout
        return r,

class VanillaRnnCell(RnnCell):

    def infer_shape(self,sx,sprev_h,sWx,sWh,sb):
        N,D = sx
        assert N == sprev_h[0]
        _,H = sprev_h
        assert sWx == (D,H)
        assert sWh == (H,H)
        assert sb == (H,)
        return (1,N,H)
    
    def forw_eval(self,x,prev_h,Wx,Wh,b):
        next_h = np.array([np.tanh(prev_h[0].dot(Wh)+x.dot(Wx)+b)])
        cache = (x,prev_h,Wx,Wh,next_h)
        return next_h,cache

    def back_eval(self,Dout,cache,bp_mask=[True]*5):
        x,prev_h,Wx,Wh,next_h = cache
        dx,dprev_h,dWx,dWh,db = bp_mask
        dz = Dout[0]*(1.-next_h[0]**2)
        if dx: dx = dz.dot(Wx.T)
        if dprev_h: dprev_h = np.array([dz.dot(Wh.T)])
        if dWx: dWx = (x.T).dot(dz)
        if dWh: dWh = (prev_h[0].T).dot(dz)
        if db: db = dz.sum(axis=0)
        return dx,dprev_h,dWx,dWh,db
    
class LstmCell_Raw(RnnCell):

    def infer_shape(self,sx,sprev_h_c,sWx,sWh,sb):
        N,D = sx
        assert sprev_h_c[0] == 2 and sprev_h_c[1] == N
        H = sprev_h_c[2]
        assert sWx == (D,4*H)
        assert sWh == (H,4*H)
        assert sb == (4*H,)
        return (2,N,H)

    def forw_eval(self,x, prev_h_c, Wx, Wh, b):
        prev_h,prev_c = prev_h_c
        z_i,z_f,z_o,z_g = np.split(x.dot(Wx)+prev_h.dot(Wh)+b,4,axis=1)
        i,f,o,g = sigmoid(z_i),sigmoid(z_f),sigmoid(z_o),np.tanh(z_g)
        next_c = f*prev_c+i*g
        tc = np.tanh(next_c)
        next_h = tc*o
        cache = (x,prev_h,prev_c,Wx,Wh,i,f,o,g,tc)
        return np.array([next_h,next_c]),cache

    def back_eval(self,Dout,cache,bp_mask=[True]*5):
        x,prev_h,prev_c,Wx,Wh,i,f,o,g,tc = cache
        dx, dprev_h_c, dWx, dWh, db = bp_mask
        N,H = prev_c.shape
        dnext_h,dnext_c = Dout
        dnext_c_total = dnext_c+dnext_h*o*(1.-tc**2)
        di,df,do,dg = dnext_c_total*g,dnext_c_total*prev_c,dnext_h*tc,dnext_c_total*i
        dz = np.zeros([N,4*H])
        dz[:,0:H],dz[:,H:2*H],dz[:,2*H:3*H],dz[:,3*H:4*H] = i*(1.-i)*di,f*(1.-f)*df,o*(1.-o)*do,(1.-g**2)*dg
        if dx: dx = dz.dot(Wx.T)
        if dprev_h_c: dprev_h_c = np.array([dz.dot(Wh.T),dnext_c_total*f])
        if dWx: dWx = x.T.dot(dz)
        if dWh: dWh = prev_h.T.dot(dz)
        if db: db = dz.sum(axis=0)
        return dx,dprev_h_c,dWx,dWh,db

class Rnn(Node):

    def init(self,rnn_cell):
        self.rnn_cell = rnn_cell

    def infer_shape(self,sx,sinit_state,sWx,sWh,sb):
        N,T,D = sx
        P,Q,H = self.rnn_cell.infer_shape(None,(N,D),sinit_state,sWx,sWh,sb)
        self.cell_state_shape = (P,Q,H)
        self.out_shape = (N,T,H)
        return (2,)#(N,T,H)

    def forw_eval(self,x,init_state,Wx,Wh,b):
        N,T,D = x.shape
        H = init_state.shape[2]
        out = np.zeros((N,T,H))
        state = init_state
        cache = []
        for t in range(T):
            state,c = self.rnn_cell.forw_eval(None,x[:,t,:],state,Wx,Wh,b)
            out[:,t,:] = state[0]
            cache.append(c)
        cache.append((T,x.shape,init_state.shape,Wx.shape,Wh.shape,b.shape))
        return np.array([out,state]),cache

    def back_eval(self,Dout,cache,bp_mask=[True]*5):
        T,shape_x, shape_state, shape_Wx, shape_Wh, shape_b = cache.pop()
        dx,dinit_state,dWx,dWh,db = bp_mask
        if dx: dx = np.zeros(shape_x)
        if dWx: dWx = np.zeros(shape_Wx)
        if dWh: dWh = np.zeros(shape_Wh)
        if db: db = np.zeros(shape_b)
        dnext_state = np.zeros(shape_state)+Dout[1]
        bp_mask[1] = True
        for t in range(T-1,-1,-1):
            dnext_state[0] += Dout[0][:,t,:]
            dx_t,dstate_t,dWx_t,dWh_t,db_t = self.rnn_cell.back_eval(None,dnext_state,cache.pop(),bp_mask) #feed gradient of next_h back in
            if bp_mask[0]: dx[:,t,:] = dx_t
            dnext_state = dstate_t
            if bp_mask[2]: dWx+=dWx_t
            if bp_mask[3]: dWh+=dWh_t
            if bp_mask[4]: db+=db_t
        if dinit_state: dinit_state=dnext_state
        return dx,dinit_state,dWx,dWh,db

    def __iter__(self):
        return iter((Projection(self,n=2,i=0,shape=self.out_shape),Projection(self,n=2,i=1,shape=self.cell_state_shape)))

class TemporalAffine(Node):

    def infer_shape(self,sx,sw,sb):
        N,T,D = sx
        assert sw[0] == D
        _,M = sw
        assert sb == (M,)
        return (N,T,M)

    def forw_eval(self,x,w,b):
        N,T,D = x.shape
        M = b.shape[0]
        out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
        cache = (x, w, b, out)
        return out, cache

    def back_eval(self,Dout,cache,bp_mask=[True]*3):
        x, w, b, out = cache
        dx,dw,db = bp_mask
        N, T, D = x.shape
        M = b.shape[0]
        if dx: dx = Dout.reshape([N * T, M]).dot(w.T).reshape([N, T, D])
        if dw: dw = Dout.reshape([N * T, M]).T.dot(x.reshape([N * T, D])).T
        if db: db = Dout.sum(axis=(0, 1))
        return dx,dw,db

class TemporalDot(Node):

    def infer_shape(self,sx,sw):
        N,T,D = sx
        assert sw[0] == D
        _,M = sw
        return (N,T,M)

    def forw_eval(self,x,w):
        N,T,D = x.shape
        _,M = w.shape
        out = x.reshape(N * T, D).dot(w).reshape(N, T, M)
        cache = (x, w, out)
        return out, cache

    def back_eval(self,Dout,cache,bp_mask=[True]*2):
        x, w, out = cache
        dx,dw = bp_mask
        N, T, D = x.shape
        _,M = w.shape
        if dx: dx = Dout.reshape([N * T, M]).dot(w.T).reshape([N, T, D])
        if dw: dw = Dout.reshape([N * T, M]).T.dot(x.reshape([N * T, D])).T
        return dx,dw

class TemporalCE(Node):

    def infer_shape(self,sx,sy,smsk):
        N,T,V = sx
        assert sy == (N,T)
        assert smsk == (N,T)
        return ()

    def forw_eval(self,x,y,msk):
        N, T, V = x.shape
        x_flat = x.reshape([N * T, V])
        y_flat = y.reshape(N * T)
        msk_flat = msk.reshape(N * T)
        probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        #loss = -np.sum(np.log(probs[np.arange(N * T), y_flat]))/(N*T) #/ N
        loss = -np.sum(msk_flat * np.log(probs[np.arange(N * T), y_flat])) / N
        cache = (probs,N,T,V,y_flat,msk_flat)
        return loss,cache

    def back_eval(self,Dout,cache,bp_mask=[True,False,False]):
        probs,N,T,V,y_flat,msk_flat=cache
        dx,dy,dmsk = bp_mask
        if dx:
            dx_flat = probs.copy()
            dx_flat[np.arange(N * T), y_flat] -= 1
            dx_flat /= N
            dx_flat *= msk_flat[:, None]
            dx = dx_flat.reshape([N, T, V])
        if dy: dy = 0
        if dmsk: dmsk = 0
        return dx,dy,dmsk
    
class Embed(Node):

    def infer_shape(self,sx,sw):
        assert len(sw) == 2
        assert len(sx) == 2
        N,T = sx
        V,D = sw #vocab size, dimensionality of embedding
        return (N,T,D)

    def forw_eval(self,x,w):
        out = w[x,:]
        V,D = w.shape
        cache = (x,V,D)
        return out,cache

    def back_eval(self,Dout,cache,bp_mask=[False,True]):
        x,V,D = cache
        dx,dw = bp_mask
        if dx: dx = 0
        if dw:
            dw = np.zeros([V,D])
            np.add.at(dw,x.flat,Dout.reshape(-1,D))
        return dx,dw

class Lstm():

    def __init__(self,num_units):
        self.num_units = num_units

    def get_zero_state(self,batch_size,learnable=False,graph=None,as_node=False):
        if as_node:
            return Variable(initializer=const_init(np.zeros([2,batch_size,self.num_units])),shape=(2,-1,self.num_units),learnable=learnable,graph=graph)
        else:
            return np.zeros([2,batch_size,self.num_units])
    
    def __call__(self,x,state):
        if not hasattr(self,'Wx'):
            self.Wx = Variable(initializer=xavier_init((x.shape[-1],self.num_units*4)),graph=x.graph)
            self.Wh = Variable(initializer=xavier_init((self.num_units,self.num_units*4)))
            self.b = Variable(initializer=const_init(np.concatenate([np.zeros(3*self.num_units),np.ones(self.num_units)])),graph=x.graph)
        if len(x.shape)==2:
            return LstmCell_Raw(x,state,self.Wx,self.Wh,self.b)
        elif len(x.shape)==3:
            return Rnn(x,state,self.Wx,self.Wh,self.b,rnn_cell=LstmCell_Raw)
        else:
            raise Exception("Invalid input shape")
