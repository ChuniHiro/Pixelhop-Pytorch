# v2019.10.25
import numpy as np 
import math
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows

def myResize(x, H, W):
    new_x = np.zeros((x.shape[0], H, W, x.shape[3]))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[3]):
            new_x[i,:,:,j] = cv2.resize(x[i,:,:,j], (W,H), interpolation=cv2.INTER_CUBIC)
    return new_x

def MaxPooling(Xinput, win=2, stride=1, padding=0):

    if padding:
        padding = win//2
        Xinput = np.pad(Xinput, ((0,0), (padding,padding), (padding,padding), (0,0)),'constant')
    ch= Xinput.shape[-1]
    X_w = view_as_windows(Xinput, (1,win,win,ch), (1,stride,stride,ch))
    X_w = abs(X_w)
    # 0 - 3 patch_idx
    # 4 - 7 patches
    X_w = X_w.max(axis=(5, 6))
    return X_w.reshape(X_w.shape[0],X_w.shape[1],X_w.shape[2],-1)

def AvgPooling(x):
    return block_reduce(x, (1, 2, 2, 1), np.mean)

def Project_concat(feature, kernelsize):
    dim = 0
    for i in range(len(feature)):
        dim += feature[i].shape[3]
        feature[i] = np.moveaxis(feature[i],0,2)
    result = np.zeros((feature[0].shape[0],feature[0].shape[1],feature[0].shape[2],dim))
    for i in range(0,feature[0].shape[0]):
        for j in range(0,feature[0].shape[1]):
            scale = 1
            boundary = 0
            for layer in range(len(feature)):
                fea = feature[layer];
                if scale == 1:
                    tmp = fea[i,j]
                else:
                    #print(i,j,i//scale,j//scale)
                    tmp = np.concatenate((tmp, fea[int( (i - boundary ) // scale),int( (j - boundary) // scale)]), axis=1)
                boundary = boundary + (kernelsize[layer] - 1) * (2 ** layer)
                scale *= 2
            result[i,j] = tmp
    result = np.moveaxis(result, 2, 0)
    return result

def Pool_Boundary_concat(feature, kernelsize):
    dim = 0
    for i in range(len(feature)):
        dim += feature[i].shape[3]
        feature[i] = np.moveaxis(feature[i],0,2)
    result = np.zeros((feature[0].shape[0],feature[0].shape[1],feature[0].shape[2],dim))
    
    for i in range(0,feature[0].shape[0]):
        for j in range(0,feature[0].shape[1]):
            scale = 1
            boundary = 0
            for layer in range(len(feature)):
                fea = feature[layer];
                # print("feature_shape",fea.shape)
                if scale == 1:
                    tmp = fea[i,j]
                else:
                    #print(i,j,i//scale,j//scale)
                    if i >= boundary and i < fea.shape[0]-boundary and j >= boundary and j < fea.shape[1]-boundary :
                        # print("check:", i, " ", j," ", boundary, " ", scale)
                        tmp = np.concatenate((tmp, fea[int( (i - boundary ) // scale),int( (j - boundary) // scale)]), axis=1)
                    else:
                        # print("check:", np.zeros(fea[i][j].shape).shape)
                        tmp = np.concatenate((tmp,  np.zeros(fea[0][0].shape)), axis=1)
                boundary = boundary + (kernelsize[layer + 1] - 1) * (2 ** (layer))
                # print("check_boundary,",boundary)
                scale *= 2
                
            result[i,j] = tmp
            print("check_tmp:", tmp.shape)
      
    result = np.moveaxis(result, 2, 0)
    return result