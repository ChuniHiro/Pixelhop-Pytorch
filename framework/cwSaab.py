# v2021.04
# A generalized version of channel wise Saab
#
# Depth goal may not achieved if no nodes's energy is larger than energy threshold or too few SaabArgs/shrinkArgs, (warning generates)
#
import numpy as np 
import pickle, gc, time
from framework.saab import Saab
# from numba import jit
# from numba import njit
from framework.layer import *
import time



def gc_invoker(func):
    def wrapper(*args, **kw):
        value = func(*args, **kw)
        gc.collect()
        # print('no sleep')
        # time.sleep(0.5)
        return value
    return wrapper

class cwSaab():
    
    def __init__(self, depth=1, TH1=0.01, TH2=0.005, SaabArgs=None, shrinkArgs=None, DCAC = False):
        self.par = {}
        self.bias = {}
        assert (depth > 0), "'depth' must > 0!"
        self.depth = (int)(depth)
        self.TH1 = TH1
        self.TH2 = TH2
        self.adaptiveTH1 = [] # record th2 from different hops
        self.DCenergy = None
        assert (SaabArgs != None), "Need parameter 'SaabArgs'!"
        self.SaabArgs = SaabArgs
        assert (shrinkArgs != None), "Need parameter 'shrinkArgs'!"
        self.shrinkArgs = shrinkArgs
        self.Energy = {}
        self.trained = False
        self.split = False
        self.transformstart = 0 # use for intermediate transform
        self.transformstop = self.depth # use for intermediate transform
        self.DCAC = DCAC
        if depth > np.min([len(SaabArgs), len(shrinkArgs)]):
            self.depth = np.min([len(SaabArgs), len(shrinkArgs)])
            print("       <WARNING> Too few 'SaabArgs/shrinkArgs' to get depth %s, \
            actual depth: %s"%(str(depth),str(self.depth)))

    # @gc_invoker
    def SaabTransform(self, X, saab, layer, train=False):
        shrinkArg, SaabArg = self.shrinkArgs[layer], self.SaabArgs[layer]
        assert ('func' in shrinkArg.keys()), "shrinkArg must contain key 'func'!"
        
        if shrinkArg['pad'] >= 1:
            
            padding = shrinkArg['win']//2
            X = np.pad(X, ((0,0), (padding,padding), (padding,padding), (0,0)),'reflect')
        
        # print('shrinkArg,', shrinkArg)
        # print('before',X.shape)
        X = shrinkArg['func'](X, shrinkArg)
        S = list(X.shape)
        # print('after',S)
        X = X.reshape(-1, S[-1])
        
        if SaabArg['num_AC_kernels'] != -1:
            S[-1] = SaabArg['num_AC_kernels']
            
        # print('check point 1')
        transformed = saab.transform(X)
        transformed = transformed.reshape(S[0],S[1],S[2],-1)
        if not self.DCAC:
            transformed = transformed[:, :, :, saab.Energy>=self.TH1] # is this wrong?
        # print("keep DC AC1?:", self.DCAC)
        # print(transformed.shape)
        
        # why doing this??
#         if train==True and self.SaabArgs[layer]['cw'] == True:
#             transformed = transformed[:, :, :, saab.Energy>=self.TH1]
            
#             print("keep checking DCenergy:", self.DCenergy)
#             print(self.adaptiveTH1)
            # threshold = max(self.DCenergy, self.TH1)
            # transformed = transformed[:, :, :, saab.Energy>=threshold]
            
            #TEST adaptive threshold to keep all DC responses
            # transformed = transformed[:, :, :, saab.Energy>=self.adaptiveTH1[-1]]
            # print('check discard here:')
            # print(saab.Energy, self.adaptiveTH1[-1])
            # print(saab.Energy>=self.adaptiveTH1[-1])
            
        return transformed
    
    # @gc_invoker
    def SaabFit(self, X, layer, bias=0):
        shrinkArg, SaabArg = self.shrinkArgs[layer], self.SaabArgs[layer]
        assert ('func' in shrinkArg.keys()), "shrinkArg must contain key 'func'!"
        
        if shrinkArg['pad'] == 1:
            
            padding = shrinkArg['win']//2
            X = np.pad(X, ((0,0), (padding,padding), (padding,padding), (0,0)),'reflect')
        
        X = shrinkArg['func'](X, shrinkArg)
        S = list(X.shape)
        X = X.reshape(-1, S[-1])
        saab = Saab(num_kernels=SaabArg['num_AC_kernels'], needBias=SaabArg['needBias'], \
                    bias=bias, useDC = SaabArg['useDC'])
        saab.fit(X)
        return saab

    # @gc_invoker
    # disabled, check below
    def discard_nodes(self, saab):
        
        """
        discard chanenl when energy is too small:
        keeping all channels and not using this function 08/31/21
        """
        
        energy_k = saab.Energy
        # print("check lowerst enregy:",energy_k[-1])
        discard_idx = np.argwhere(energy_k<self.TH2)
        saab.Kernels = np.delete(saab.Kernels, discard_idx, axis=0) 
        saab.Energy = np.delete(saab.Energy, discard_idx)
        saab.num_kernels -= discard_idx.size
        # print("check discard_nodes with TH2=", self.TH2)
        return saab

    # @gc_invoker
    # @jit(parallel = True, fastmath = True)
    def cwSaab_1_layer(self, X, train, bias=None):
        if train == True:
            saab_cur = []
            bias_cur = []
        else:
            saab_cur = self.par['Layer'+str(0)]
            # bias_cur = self.bias['Layer'+str(0)]
        transformed, eng, DC = [], [], []
        
        if self.SaabArgs[0]['cw'] == True:
            S = list(X.shape)
            S[-1] = 1
            X = np.moveaxis(X, -1, 0)
            leni = X.shape[0]
#             print("check leni:", leni)
            for i in range(leni):
                X_tmp = X[i].reshape(S)
                if train == True:
                    saab = self.SaabFit(X_tmp, layer=0)
                    # print("find DC energy here")
                    # print(saab.Energy)
                    # self.adaptiveTH1.append(saab.Energy[0])
                    if not self.DCAC:
                        saab = self.discard_nodes(saab)
                    saab_cur.append(saab)
                    # bias_cur.append(saab.Bias_current)#*np.ones(saab.Energy.size))
                    eng.append(saab.Energy)
                    # print("layer0 ENERGY in cw @hop0, not implemented yet!")
                    self.DCenergy = saab.Energy[-1]
                    # print(len(saab.Energy),saab.Energy)
                    transformed.append(self.SaabTransform(X_tmp, saab=saab, layer=0, train=True))
                else:
                    if len(saab_cur) == i:
                        break
                    transformed.append(self.SaabTransform(X_tmp, saab=saab_cur[i], layer=0))
            transformed = np.concatenate(transformed, axis=-1)
        else:
            
            if train == True:
                saab = self.SaabFit(X, layer=0)
                saab = self.discard_nodes(saab)
                saab_cur.append(saab)
                bias_cur.append(saab.Bias_current)#*np.ones(saab.Energy.size))
                eng.append(saab.Energy)
                print("check energy:", saab.Energy)
                transformed = self.SaabTransform(X, saab=saab, layer=0, train=True)
            else:
                transformed = self.SaabTransform(X, saab=saab_cur[0], layer=0)
                
        if train == True:
            self.par['Layer0'] = saab_cur
            # self.bias['Layer'+str(0)] = bias_cur
            self.Energy['Layer0'] = eng
          
        # print("check adaptive energy", self.adaptiveTH1)
        return transformed

    # @gc_invoker
    # @jit(parallel = True, fastmath = True)
    def cwSaab_n_layer_train(self, X, train, layer):
        output, eng_cur, DC, ct, pidx = [], [], [], -1, 0
        self.DCenergy=1
        S = list(X.shape)
        saab_prev = self.par['Layer'+str(layer-1)]
        saab_cur = []
        
        # if channel wise saab
        if self.SaabArgs[layer]['cw'] == True:
            
            S[-1] = 1
            X = np.moveaxis(X, -1, 0)
            # print("debug ct: X shape")
            # print(X.shape)
            print("in cwsaab_n_layer, current #Saab:", len(saab_prev))
            print("check input X:",X.shape)
            for i in range(len(saab_prev)):

                ct = -1
                print("in channel", i, "Energy:", saab_prev[i].Energy.shape[0])
                for j in range(saab_prev[i].Energy.shape[0]):
                    
                    if self.DCAC:
                        
                        ct += 1
                        if ct >= 2:
                            break
                    else:
                        
                        if saab_prev[i].Energy[j] < self.TH1:
                            # it is a leaf node
                            continue
                        else: 
                            # intermediate node
                            # need further splitting
                            ct += 1
                        
                    self.split = True
                    if self.DCAC:
                        X_tmp = X[ct + i * 10].reshape(S)
                    else:   
                        X_tmp = X[ct].reshape(S)
                    # print("check energy:",ct,saab_prev[i].Energy[j])
                    saab = self.SaabFit(X_tmp, layer=layer)
#                     saab = self.SaabFit(X_tmp, layer=layer, bias=bias_prev[i])
                    saab.Energy *= saab_prev[i].Energy[j]
                    if not self.DCAC:
                        saab = self.discard_nodes(saab) # discard nodes with E < th2
                    saab_cur.append(saab)
                    # no bias 08/31/21
#                     bias_cur.append(saab.Bias_current)#*np.ones(saab.Energy.size))
                    eng_cur.append(saab.Energy) 
            
                    '''Clean the Cache'''
                    X_tmp = None
                    gc.collect()
            
            #inference here!
            ct = -1
            pidx = 0
#             print("after all saab DCenergy=", self.DCenergy)
            
#             print("debug saab transform:",len(saab_prev), len(saab_cur))
#             print(X.shape)
            for i in range(len(saab_prev)):
                
                ct = -1
#                 print("debug saab transform energy shape:", saab_prev[i].Energy.shape[0],\
#                       'for previous channel',i)
                for j in range(saab_prev[i].Energy.shape[0]):
                    
                    if self.DCAC:
                        
                        ct += 1
                        if ct >= 2:
                            break
                    else:
                        # only transform intermediate nodes
                        if saab_prev[i].Energy[j] < self.TH1:
                            # it is a leaf node
                            # discardchannel += 1
    #                         print("discarded:",saab_prev[i].Energy[j], self.adaptiveTH1[-1])
                            continue
                        else:
                            # intermediate node
                            ct += 1
                    if self.DCAC:
                        X_tmp = X[ct + i * 10].reshape(S)
                    else:
                        X_tmp = X[ct].reshape(S)
#                     print(ct, pidx)
                    out_tmp = self.SaabTransform(X_tmp, saab=saab_cur[pidx], layer=layer, train=True)
#                     print('check output_tmp:',out_tmp.shape)
                    pidx += 1
                    output.append(out_tmp)
                    
                    
                    out_tmp = None
                    gc.collect()
            
            output = np.concatenate(output, axis=-1)
            # self.adaptiveTH1.append(self.DCenergy)
                    
        else:
            """
            check consistency:
            this energy is not multipled by previous energy channelwisely
            """
            saab = self.SaabFit(X, layer=layer)
            saab = self.discard_nodes(saab)
            saab_cur.append(saab)
            output = self.SaabTransform(X, saab=saab, layer=layer, train=True)

        print("in cwsaab_layer_",layer)
        # print("discard channels",discardchannel)
        print("output:",output.shape)
        # print("output:",len(output), output[0].shape, "dimension:", len(output) * output[0].shape[-1])


        if self.split == True or self.SaabArgs[0]['cw'] == False: # ??

            # print("debug:", self.split, self.SaabArgs[0]['cw'] )
            self.par['Layer'+str(layer)] = saab_cur
#             self.bias['Layer'+str(layer)] = bias_cur
            self.Energy['Layer'+str(layer)] = eng_cur

        # print(eng_cur)
        # eng_cur.sort(reverse = True)
            
        # self.adaptiveTH1 = eng_cur
        # print("check adaptive energy", self.adaptiveTH1)
        return output

    # @jit(parallel = True, fastmath = True)
    def cwSaab_n_layer_trans(self, X, layer):
        output, eng_cur, DC, ct, pidx = [], [], [], -1, 0
        S = list(X.shape)
        saab_prev = self.par['Layer'+str(layer-1)]
#         bias_prev = self.bias['Layer'+str(layer-1)]
        # discardchannel = 0
        saab_cur = self.par['Layer'+str(layer)]
#         print('Layer'+str(layer), len(saab_cur))
        
        
        if self.SaabArgs[layer]['cw'] == True:
            S[-1] = 1
            X = np.moveaxis(X, -1, 0)
#             print(X.shape)
            # t0 = time.time()
            # print('time for loop here:', X.shape)
            saab_prev_len = len(saab_prev)
            # print("check saab_prev_len", saab_prev_len)
            for i in range(saab_prev_len): # for each channel in previous hop
                
                lenj = saab_prev[i].Energy.shape[0]
                ct = -1
#                 print("for i = ", i, "lenj",lenj)
                for j in range(lenj): # 
                    
                    # print('test')
                    # print(saab_prev[:])
                    # if don't discard here, not in consistend with saab_cur
#                     print("check all recorded tH:", self.adaptiveTH1)
#                     print("check discarding:", saab_prev[i].Energy[j], self.adaptiveTH1[layer-1], layer)
                    # Bug fixed : this should be the threshold from previous layer!!
                    # threshold = max(self.adaptiveTH1[layer-1], self.TH1)
                    
                    # print('saab_prev[i].Energy[j]', i, j , saab_prev[i].Energy[j])
                    if self.DCAC:
                        
                        ct += 1
                        if ct >= 2:
                            break
                    else:
                        
                        if saab_prev[i].Energy[j] < self.TH1:
                            # it is a leaf node
    #                         discardchannel += 1
                            continue
                        else:
                            ct+=1

                        
                    self.split = True
                    if self.DCAC:
                        X_tmp = X[ct + i * 10].reshape(S)
                    else:
                        X_tmp = X[ct].reshape(S)
                    out_tmp = self.SaabTransform(X_tmp, saab=saab_cur[pidx], layer=layer)
                    pidx += 1
                    output.append(out_tmp)
                    
            output = np.concatenate(output, axis=-1)
                    
        else:
            output = self.SaabTransform(X, saab=saab_cur[0], layer=layer)
        
        return output
    
    # @jit(parallel = True, fastmath = True)
    def fit(self, X):

        print('=' * 5 + '>c/w Saab Train Hop 0')
        X = self.cwSaab_1_layer(X, train=True)
        
        if self.shrinkArgs[0]['pooling'] != 0:
            # X = MaxPooling(X)
            win, stride, padding = self.shrinkArgs[i]['poolingParms']
            X = MaxPooling(X, win, stride, padding)
        S= list(X.shape)
        print('FEATURE SHAPE in Hop0',S)
        
        for i in range(1, self.depth):
            
            print('=' * 45 + f'>c/w Saab Train Hop {i}')
            print("check input featmap:", X.shape)
            X = self.cwSaab_n_layer_train(X, train = True, layer = i)
            if self.shrinkArgs[i]['pooling'] != 0:

                # X = MaxPooling(X)
                win, stride, padding = self.shrinkArgs[i]['poolingParms']
                X = MaxPooling(X, win, stride, padding)
            S= list(X.shape)
            print('CHECK FEATURE SHAPE',S)
            if (self.split == False and self.SaabArgs[i]['cw'] == True):
                self.depth = i
                print("       <WARNING> Cannot futher split, actual depth: %s"%str(i))
                break
            
            self.split = False
        self.trained = True
    
    # @jit(parallel = True, fastmath = True)
    def transform(self, X, starthop = 0, stophop = None):
        assert (self.trained == True), "Must call fit first!"
        """
        collect results after Maxpooling!
        """
        output, DC = [], []
        
        # t0 = time.time()
        if starthop == 0:
            
            X = self.cwSaab_1_layer(X, train = False)
            # t1 = time.time()
            if self.shrinkArgs[0]['pooling'] != 0:
                # X = MaxPooling(X)
                win, stride, padding = self.shrinkArgs[i]['poolingParms']
                X = MaxPooling(X, win, stride, padding)
            # S= list(X.shape)
            # print("layer 1", t1-t0)
            # print('HOP 0, CHECK FEATURE SHAPE',S)
            output.append(X)
            starthop += 1
        
        if stophop == None:
            
            stophop = self.depth
            
#         print("check X from Hop0:",X.shape)
        for i in range(starthop, stophop):
            # X = self.cwSaab_n_layer(X, train=False, layer=i)
            X = self.cwSaab_n_layer_trans(X, layer=i)
            
            if self.shrinkArgs[i]['pooling'] != 0:
                # X = MaxPooling(X)
                win, stride, padding = self.shrinkArgs[i]['poolingParms']
                X = MaxPooling(X, win, stride, padding)
            
            # t2 = time.time()
            
            # S= list(X.shape)
            # print('HOP', str(i), 'time:',t2-t1, 'CHECK FEATURE SHAPE',S)
            output.append(X)
            
        return output
    
    def transform_singleHop(self, X, layer=0):
        assert (self.trained == True), "Must call fit first!"
        if layer==0:
            output = self.cwSaab_1_layer(X, train = False)
        else:
            output = self.cwSaab_n_layer_trans(X, train=False, layer=layer)
        return output
    
