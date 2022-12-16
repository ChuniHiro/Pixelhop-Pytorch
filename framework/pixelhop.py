# v2021.04
import numpy as np 
from framework.cwSaab import cwSaab
import pickle

class Pixelhop(cwSaab):
    def __init__(self, depth=1, TH1=0.005, TH2=0.001, SaabArgs=None, shrinkArgs=None, concatArg=None):
        super().__init__(depth=depth, TH1=TH1, TH2=TH2, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs)
        self.TH1 = TH1
        #how to get an adaptive TH2? -> in cwSaab!
        self.TH2 = TH2
        
        self.idx = []        
        self.concatArg = concatArg

    def fit(self, X):
        super().fit(X)
        return self

    def transform(self, X, starthop=0, stophop=None):
        X = super().transform(X, starthop, stophop)
        return self.concatArg['func'](X, self.concatArg)

    def transform_singleHop(self, X, layer=0):
        X = super().transform_singleHop(X, layer=layer)
        return X

    '''Methods for Saving & Loading'''
    def save(self, filename: str):
        assert (self.trained == True), "Need to Train First"
        par = {}
        par['kernel'] = self.par
        par['depth'] = self.depth
        par['energyTH'] = self.energyTH
        par['energy'] = self.Energy
        par['SaabArgs'] = self.SaabArgs
        par['shrinkArgs'] = self.shrinkArgs
        par['concatArgs'] = self.concatArg
        par['concatArg_pixel2'] = self.concatArg
        par['TH1'] = self.TH1
        par['TH2'] = self.TH2

        with open(filename + '.pkl','wb') as f:
            pickle.dump(par, f)
        return

    def load(self, filename: str):
        par = pickle.load(open(filename + '.pkl','rb'))
        self.par = par['kernel']
        self.depth = par['depth']
        self.energyTH = par['energyTH']
        self.Energy = par['energy']
        self.SaabArgs = par['SaabArgs']
        self.shrinkArgs = par['shrinkArgs']
        self.concatArg = par['concatArgs']
        self.trained = True

        self.concatArg = par['concatArg_pixel2']
        self.TH1 = par['TH1']
        self.TH2 = par['TH2']
        
        return self


