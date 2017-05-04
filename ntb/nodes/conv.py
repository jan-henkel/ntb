from .nodes import Node
import numpy as np

def im2col_aux(shape_x,shape_w,stride,pad):
    N,C,H,W=shape_x
    F,_,HH,WW=shape_w
    H_pad = H+pad[0]
    W_pad = W+pad[1]
    H_out = (H_pad-HH)//stride+1
    W_out = (W_pad-WW)//stride+1
    shape_padded = (N,C,H_pad,W_pad)
    shape_2d = (C * HH * WW, N * H_out * W_out)
    shape_6d = (C, HH, WW, N, H_out, W_out)
    strides_6d = (H_pad * W_pad, W_pad, 1, C * H_pad * W_pad, stride * W_pad, stride)
    return F,N,H_out,W_out,shape_padded,shape_2d,shape_6d,strides_6d

def as_strided(x,shape,strides):
    strides=np.array(strides)*x.itemsize
    return np.lib.stride_tricks.as_strided(x=x,shape=shape,strides=strides)

def pad_4d(x,pad,c=0):
    if c==0:
        return np.pad(x,((0, 0), (0, 0), (pad[0]//2, (pad[0]+1)//2), (pad[1]//2, (pad[1]+1)//2)), mode='constant')
    else:
        return np.pad(x,((0, 0), (0, 0), (pad[0]//2, (pad[0]+1)//2), (pad[1]//2, (pad[1]+1)//2)), mode='constant',constant_values=(c,))

def unpad_4d(x,pad):
    return x[:,:,pad[0]//2:x.shape[2]-(pad[0]+1)//2,pad[1]//2:x.shape[3]-(pad[1]+1)//2]

def get_pad(H,W,HH,WW,stride,padding_type='minimal'):
    if padding_type=='minimal':
        return (-(H-HH) % stride,-(W-WW) % stride)
    else:
        return (HH + (-H % stride),WW + (-W % stride))
    
def im2col(x,shape_2d,shape_6d,strides_6d,pad):
    x_padded = pad_4d(x,pad)
    x_stride = as_strided(x_padded,shape=shape_6d,strides=strides_6d)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = shape_2d
    return x_cols

def col2im(x_cols,shape_padded,shape_6d,strides_6d,pad):
    x_im_padded = np.zeros(shape_padded)
    x_im_cols = as_strided(x_im_padded,shape=shape_6d,strides=strides_6d)
    np.add.at(x_im_cols,slice(None),x_cols.reshape(shape_6d))
    x_im = unpad_4d(x_im_padded,pad)
    return x_im

class Conv2d(Node):

    #Conv2d takes the following inputs:
    #An input tensor x of shape (N,C,H,W) (batchsize, channels, height, width)
    #A tensor w of filters, shape (F,C,HH,WW) (number of filters, input channels, height, width)
    #A tensor of biases, shape (F,)
    #Furthermore, stride and padding need to be passed as keyword arguments 'stride' and 'pad'
    #stride is a scalar, pad a pair of values, one for each axis (distributed as evenly as possible on the boundaries)
    
    def init(self,**kwargs):
        self.stride,self.pad = kwargs['stride'], kwargs['pad']

    def infer_shape(self,sx,sw,sb):
        N,C,H,W = sx
        assert C == sw[1]
        F,_,HH,WW = sw
        assert (F,) == sb
        H += self.pad[0]
        W += self.pad[1]
        assert (W - WW) % self.stride == 0, 'width does not work'
        assert (H - HH) % self.stride == 0, 'height does not work'
        Hout = (H-HH)//self.stride+1
        Wout = (W-WW)//self.stride+1
        return (N,F,Hout,Wout)
        
    def forw_eval(self,x,w,b):
        F,N,H_out,W_out,shape_padded,shape_2d,shape_6d,strides_6d = im2col_aux(x.shape,w.shape,self.stride,self.pad)
        x_cols = im2col(x,shape_2d,shape_6d,strides_6d,self.pad)
        res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)
        res.shape = (F, N, H_out, W_out)
        out = res.transpose(1, 0, 2, 3)
        out = np.ascontiguousarray(out)
        cache = (x_cols,w,F,shape_padded,shape_6d,strides_6d)
        return out,cache

    def back_eval(self,Dout,cache,bp_mask=[True]*3):
        x_cols,w,F,shape_padded,shape_6d,strides_6d=cache
        dx,dw,db = bp_mask
        Dout_reshaped = Dout.transpose(1, 0, 2, 3).reshape(F, -1)
        if dx:
            dx_cols = w.reshape(F, -1).T.dot(Dout_reshaped)
            dx = col2im(dx_cols,shape_padded,shape_6d,strides_6d,self.pad)
        if dw: dw = Dout_reshaped.dot(x_cols.T).reshape(w.shape)
        if db: db = np.sum(Dout, axis=(0, 2, 3))
        return dx,dw,db

class MaxPool(Node):
    
    def init(self,**kwargs):
        self.pool_size = kwargs['pool_size']
        self.pad = kwargs['pad']

    def infer_shape(self,sx):
        N,C,H,W=sx
        H+=self.pad[0]
        W+=self.pad[1]
        assert H % self.pool_size[0] == 0
        assert W % self.pool_size[1] == 0
        return (N,C,H//self.pool_size[0],W//self.pool_size[1])
                    
    def forw_eval(self,x):
        N, C, H, W = x.shape
        H+=self.pad[0]
        W+=self.pad[1]
        x_padded = pad_4d(x,self.pad,-np.inf)
        x_reshaped = x_padded.reshape(N, C, H // self.pool_size[0], self.pool_size[0], W // self.pool_size[1], self.pool_size[1])
        out = x_reshaped.max(axis=3).max(axis=4)
        cache = (x_padded.shape, x_reshaped, out)
        return out, cache

    def back_eval(self,Dout,cache,bp_mask=[True]):
        dx, = bp_mask
        if dx:
            shape_padded, x_reshaped, out = cache
            dx_reshaped = np.zeros_like(x_reshaped)
            out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
            mask_eq = (x_reshaped == out_newaxis)
            Dout_newaxis = Dout[:, :, :, np.newaxis, :, np.newaxis]
            Dout_broadcast, _ = np.broadcast_arrays(Dout_newaxis, dx_reshaped)
            dx_reshaped[mask_eq] = Dout_broadcast[mask_eq]
            dx_reshaped /= np.sum(mask_eq, axis=(3, 5), keepdims=True)
            dx_padded = dx_reshaped.reshape(shape_padded)
            dx = unpad_4d(dx_padded,self.pad)
        return dx,
    
