# v2021.04

# Saab transformation
# modeiled from https://github.com/davidsonic/Interpretable_CNN
import time
import numpy as np
# import numba
from sklearn.decomposition import PCA, IncrementalPCA
import scipy.linalg.blas as blas
# import os
from scipy.linalg import get_blas_funcs
import matplotlib.pyplot as plt

# @numba.jit(nopython = True, parallel = True)
def pca_cal(X: np.ndarray):
    
    cov = X.transpose() @ X
    eva, eve = np.linalg.eigh(cov)

    # max_abs_cols = np.argmax(np.abs(eve), axis = 0)
    # signs = np.sign(eve[max_abs_cols, range(eve.shape[1])])
    # eve *= signs

    inds = eva.argsort()[::-1]
    eva = eva[inds]
    eva = np.absolute(eva)
    # print("check eigen value:", eva[-1])
    kernels = eve.transpose()[inds]
    return kernels, eva / (X.shape[0] - 1)

# @numba.jit(forceobj = True, parallel = True)
def remove_mean(X: np.ndarray, feature_mean: np.ndarray):
    return X - feature_mean

# @numba.njit(nopython = True, parallel = True)
# @numba.njit(parallel = True)
def feat_transform(X: np.ndarray, kernel: np.ndarray):
    # return np.matmul(X, kernel.transpose())
    return X @ kernel.transpose()


class Saab():

    def __init__(self, num_kernels=-1, needBias=True, bias=0,  useDC=1):
        
        self.par = None
        self.Kernels = []
        self.Bias_previous = bias # bias calculated from previous
        self.Bias_current = []
        self.Mean0 = []
        self.Energy = []
        self.num_kernels = num_kernels
        self.useDC = useDC
        self.needBias = needBias
        self.trained = False
        # self.DCEnergy = []


    # @numba.jit(forceobj = True, parallel = True)
    def fit(self, X): 
        """
        ver 07/15/21:
        2 steps:
        1. AC/DC Seperation
        2. PCA on all DC components

        No need of Bias
        No need of Nomalization
        """
        assert (len(X.shape) == 2), "Input must be a 2D array!"
#         print('check num_kernels', self.num_kernels, X.shape)
        if self.num_kernels == -1:
            self.num_kernels = X.shape[-1]
            
            
        # X = X.astype('float32')
        # add bias
        # if self.needBias == True:
        #     X += self.Bias_previous
       
        if self.useDC == False:
            
#             print(X.shape)
#             X = X.astype('float32')
#             kernels, eva = pca_cal(X)
#             energy = eva / np.sum(eva)
    
            # remove feature mean --> self.Mean0 and store it!!
            self.Mean0 = np.mean(X, axis = 0, keepdims = True)
            X = remove_mean(X, self.Mean0)
            
            # batchsize = min(X.shape[0]//10, 400* 448 * 448)
            # print("Using IPCA, Batchsize=", batchsize)
            # pca = IncrementalPCA(n_components=self.num_kernels, batch_size = batchsize ).fit(X)
            
            # print("Using PCA")
            pca = PCA(n_components=self.num_kernels, svd_solver = 'full' ).fit(X)
            
            kernels = pca.components_
            energy = pca.explained_variance_ / np.sum(pca.explained_variance_)
            print("PQR energy:", energy)

        else: 
             # step 1 : DC/ AC seperation
            # remove DC, get AC components
            dc = np.mean(X, axis = 1, keepdims = True)
            X = remove_mean(X, dc) #now X is AC components

            # calcualte bias
            # self.Bias_current = np.max(np.linalg.norm(X, axis=1)) #* 1 / np.sqrt(X.shape[1])
            # remove feature mean --> self.Mean0 and store it!!
            self.Mean0 = np.mean(X, axis = 0, keepdims = True)
            X = remove_mean(X, self.Mean0)
            
            # """ full normalization """
            # X /=  np.std(X, axis=0)
            '''Rewritten PCA Using Numpy'''
    
            # step 2: Do PCA on AC
            # kernels, eva = pca_cal(X)
            # use SKlearn Incremental Batch process
            # datasize = X.shape[0]
            # print("datasize:", datasize)
            # batchsizemax = 500 * 448 * 448
            # if datasize < batchsizemax:
                
            # print("Using PCA")
            pca = PCA(n_components=self.num_kernels, svd_solver = 'full' ).fit(X)
            
            # else:
            #     batchsize = min(X.shape[0]//5, batchsizemax)
            #     # print("Using IPCA, Batchsize=", batchsize)
            #     pca = IncrementalPCA(n_components=self.num_kernels, batch_size = batchsize ).fit(X)
            # ## incrementalPCA && PCA will get zero mean
            
            # print("Using PCA")
            # pca = PCA(n_components=self.num_kernels, svd_solver = 'full' ).fit(X)
            kernels = pca.components_
            eva = pca.explained_variance_ 
            
            largest_ev = np.var(dc * np.sqrt(X.shape[-1]))  
    
            # before 07/15/21
            # dc_kernel = 1 / np.sqrt(X.shape[-1]) * np.ones((1, X.shape[-1]))# / np.sqrt(largest_ev)
            # kernels = np.concatenate((dc_kernel, kernels[:-1]), axis = 0)
            # energy = np.concatenate((np.array([largest_ev]), eva[:-1]), axis = 0)
            # energy = energy / np.sum(energy)
    
            # ver 07/15/21
            dc_kernel = 1 / np.sqrt(X.shape[-1]) * np.ones((1, X.shape[-1]))
            kernels = np.concatenate((dc_kernel, kernels), axis = 0)
            energy = np.concatenate((np.array([largest_ev]), eva), axis = 0)
            energy = energy / np.sum(energy)
            # print("check energy:", energy[-1])
            # print("debug:")
            # print("dc_kernel", dc_kernel.shape)
            # print(X.shape[-1], dc_kernel)
            # print("kernels", kernels.shape)
            # print(kernels)
            # print("energy", energy[-1])
            
        # plt.plot(energy)
        # plt.show()
        # print(energy)
        
        self.Kernels, self.Energy = kernels.astype('float32'), energy
        self.trained = True

    # @numba.jit(forceobj = True, parallel = True)
    def transform(self, X):
        """
        0805 update:
            -during transforms, calculate dc -> remove dc -> remove global mean0 (zero mean)
            -the process should be in accordance with fit
            -definition of DC: mean of all samples
            
            -discard Bias (largest ac response)
        """
        assert (self.trained == True), "Must call fit first!"
        
        if self.useDC == False:
            
            # ZERO MEAN FIRST
            # print(self.Mean0)
            X = remove_mean(X, self.Mean0)
            X = np.matmul(X, self.Kernels.transpose()).astype('float32')
            
        else:
            # X = X.astype('float32')
            # if self.needBias == True:
            #     X += self.Bias_previous
            
            
            # print('using numba') # NOT TRANSPOSING?? Already tranposed in pca_cal!!! 
            # X = feat_transform(X, self.Kernels)
            
            # dc is local patch mean
            dc = np.mean(X, axis = 1, keepdims = True)
            # print("Kernels")
            # print(self.Kernels)
            # print("dc\n", X.shape, dc.shape)
    
            X = remove_mean(X, dc) #now X is AC components
            # dc = np.matmul(dc, self.Kernels[0].transpose()).astype('float32')
            dc = (dc / np.sqrt(X.shape[-1])).astype('float32')
            # print("check dc:",dc.shape)
            # ac = np.matmul(X, self.Kernels[1:].transpose()).astype('float32')
            
            X = remove_mean(X, self.Mean0).astype('float32')
            # kernels will be transposed in feat_transform
            ac = feat_transform(X, self.Kernels[1:]).astype('float32')
            # ac = feat_transform(X, self.Kernels[1:].transpose()).astype('float32')
            X = np.concatenate([dc, ac], axis=1)
            
        return X

