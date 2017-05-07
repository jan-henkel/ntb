import matplotlib.pyplot as pp
import numpy as np
from itertools import product

def preprocess_images(X):
    if len(X.shape)==4:
        X = X.transpose(0,2,3,1)
        if X.shape[-1] == 1:
            X = X.reshape(X.shape[:-1])
    elif len(X.shape)==2:
        w = int(np.sqrt(X.shape[1]))
        assert w*w==X.shape[1],'Could not guess image dimensions'
        X = X.reshape(-1,w,w)
    return X.astype('uint8')

def preprocess_image(x):
    if len(x.shape)==3:
        x = x.transpose(1,2,0)
        if x.shape[-1] == 1:
            x = x.reshape(x.shape[:-1])
    elif len(x.shape)==1:
        w = int(np.sqrt(x.shape[0]))
        assert w*w==x.shape[0],'Could not guess image dimensions'
        x = x.reshape(w,w)
    return x.astype('uint8')

def render_images(X,y,classes,samples_per_class=7,**kwargs):
    pp.rcParams['image.interpolation'] = kwargs.get('interpolation','nearest')
    pp.rcParams['figure.figsize'] = kwargs.get('figsize',(5.0, 5.0))
    pp.rcParams['image.cmap'] = kwargs.get('cmap','gray')
    num_classes = len(classes)
    fig, axarr = pp.subplots(samples_per_class,num_classes)
    for c, cls in enumerate(classes):
        idxs = np.flatnonzero(y == c)
        idxs = np.random.choice(idxs, samples_per_class, replace=True)
        for i, idx in enumerate(idxs):
            axarr[i,c].axis('off')
            axarr[i,c].imshow(preprocess_image(X[idx]))
            if i == 0:
                axarr[i,c].set_title(cls)
    pp.show()

def render_images_1(X,y,classes,samples_per_class=7,**kwargs):
    pp.rcParams['image.interpolation'] = kwargs.get('interpolation','nearest')
    pp.rcParams['figure.figsize'] = kwargs.get('figsize',(5.0, 5.0))
    pp.rcParams['image.cmap'] = kwargs.get('cmap','gray')
    num_classes = len(classes)
    
    for c, cls in enumerate(classes):
        idxs = np.flatnonzero(y == c)
        idxs = np.random.choice(idxs, samples_per_class, replace=True)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + c + 1
            pp.subplot(samples_per_class, num_classes, plt_idx)
            pp.imshow(preprocess_image(X[idx]))
            pp.axis('off')
            if i == 0:
                pp.title(cls)
    pp.show()

def render_predictions(X,y,y_pred,classes,samples_per_class=7,correct=False,**kwargs):
    pp.rcParams['image.interpolation'] = kwargs.get('interpolation','nearest')
    pp.rcParams['figure.figsize'] = kwargs.get('figsize',(5.0, 5.0))
    pp.rcParams['image.cmap'] = kwargs.get('cmap','gray')
    num_classes = len(classes)
    for c, cls in enumerate(classes):
        idxs = np.flatnonzero((y_pred == c) & ((y == y_pred) == correct))
        try:
            idxs = np.random.choice(idxs, samples_per_class, replace=True)
        except:
            continue
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + c + 1
            pp.subplot(samples_per_class, num_classes, plt_idx)
            pp.imshow(preprocess_image(X[idx]))
            pp.axis('off')
            if i == 0:
                pp.title(cls)
    pp.show()


    
def render_confusion_sample(data,model=None,pred_fn=None,subset='val',**kwargs):
    try:
        predict = model.predict
    except:
        predict = pred_fn
    pp.rcParams['image.interpolation'] = kwargs.get('interpolation','nearest')
    pp.rcParams['figure.figsize'] = kwargs.get('figsize',(5.0, 5.0))
    pp.rcParams['image.cmap'] = kwargs.get('cmap','gray')
    X,y = data['X_'+subset],data['y_'+subset]
    y_pred = predict(X)
    num_classes = len(data['classes'])
    fig, axarr = pp.subplots(num_classes+1,num_classes+1)
    for a in axarr.flat:
        a.axis('off')
    for c,cls in enumerate(data['classes']):
        axarr[0,c+1].annotate(cls,(0.1,0.1),xycoords='axes fraction')
        axarr[c+1,0].annotate(cls,(0.1,0.1),xycoords='axes fraction')
    for correct in range(num_classes):
        idxs = np.flatnonzero(y == correct)
        X_ = X[idxs]
        y_pred_ = y_pred[idxs]
        for pred in range(num_classes):
            idxs_ = np.flatnonzero(y_pred_ == pred)
            try:
                idx = np.random.choice(idxs_)
            except:
                continue
            axarr[correct+1,pred+1].imshow(preprocess_image(X_[idx]))
