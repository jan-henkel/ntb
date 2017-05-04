import numpy as np
from .vis import render_images

def load(mode="1d",num_train=2000,num_val=500,path = './ntb/datasets/mnist/'):
    label_header_dt = np.dtype([('magic_number', '>i4', 1),
                                ('num_images', '>i4', 1)])
    img_header_dt = np.dtype([('magic_number', '>i4', 1),
                              ('num_images', '>i4', 1),
                              ('num_rows', '>i4', 1),
                              ('num_cols', '>i4', 1)])
    f = open(path+'train-labels-idx1-ubyte', 'rb')
    [label_header] = np.fromfile(f, label_header_dt, 1)
    labels = np.fromfile(f, np.dtype('uint8'),
                         label_header['num_images'])
    f = open(path+'train-images-idx3-ubyte', 'rb')
    [img_header] = np.fromfile(f, img_header_dt, 1)
    if mode=="1d":
        images = np.fromfile(f, np.dtype(('uint8', (img_header['num_rows'] * img_header['num_cols'],))), img_header['num_images'])
    elif mode=="2d":
        images = np.fromfile(f, np.dtype(('uint8', (1,img_header['num_rows'],img_header['num_cols']))), img_header['num_images'])
    perm = np.random.permutation(images.shape[0])
    images = images[perm]
    labels = labels[perm]
    data = {"X_train":images[:num_train],"y_train":labels[:num_train],"X_val":images[num_train:num_train+num_val],"y_val":labels[num_train:num_train+num_val],"classes":[str(i) for i in range(10)]}
    def visualize(data=data,subset="train"):
        X,y,classes = data["X_"+subset],data["y_"+subset],data["classes"]
        render_images(X,y,classes,10,cmap='gray_r',figsize=(7.,7.))
    return data,visualize
