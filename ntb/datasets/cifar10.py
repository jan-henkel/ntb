import numpy as np
import pickle
import os
from .vis import render_images

def load(num_train=50000,num_val=10000,path='./ntb/datasets/cifar-10-batches-py/'):
    n = 0
    Xs = list()
    ys = list()
    
    def loadXy_(filename):
        with open(filename,'rb') as f:
            datadict = pickle.load(f,encoding='latin1')
        return datadict['data'].reshape([-1,3,32,32]),np.array(datadict['labels'])
    
    for i in range(5):
        X_,y_ = loadXy_(os.path.join(path,'data_batch_'+str(i+1)))
        Xs.append(X_)
        ys.append(y_)
        n+=X_.shape[0]
        if(n>num_train):
            break
        
    X_train = np.concatenate(Xs)[:num_train]
    y_train = np.concatenate(ys)[:num_train]

    X_val, y_val = loadXy_(os.path.join(path,'test_batch'))
    X_val = X_val[:num_val]
    y_val = y_val[:num_val]
    
    with open(os.path.join(path,'batches.meta'),'rb') as f:
        class_dict = pickle.load(f,encoding='latin1')
        classes = class_dict['label_names']
    
    data = {'X_train':X_train,'y_train':y_train,'X_val':X_val,'y_val':y_val,'classes':classes}
    def visualize(data=data,subset='train'):
        X,y,classes = data['X_'+subset],data['y_'+subset],data['classes']
        render_images(X,y,classes,figsize=(11.,7.))
    return data,visualize


