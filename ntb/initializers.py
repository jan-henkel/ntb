import numpy as np

def normal_init(mean=0.,std=1.,shape=()):
    def _init():
        return np.clip(np.random.normal(mean,std,shape),-4*std,+4*std)
    return _init

def xavier_init(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    std=np.sqrt(2. / (fan_in+fan_out))
    def _init():
        return np.clip(np.random.normal(0.,std,shape),-4*std,+4*std)
    return _init

def const_init(c,shape=None):
    if shape is None:
        def _init():
            return np.array(c)
    else:
        assert np.shape(c) == (), "Not a scalar"
        def _init():
            return (c*np.ones(shape))
    return _init
