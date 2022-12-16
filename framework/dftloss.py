import numpy as np
from collections import Counter

def split_process_we(X, y, split, numclass = 2):

    leftidx = np.where(X<=split)
    rightidx = np.where(X>split)
    
    # print("check left and right:", leftidx[0].shape, rightidx[0].shape)
    return weighted_entropy_for_splitting(leftidx, rightidx, y, numclass)

def cal_entropy(prob,numclass=2):
    
    prob_tmp = np.copy(prob)
    prob_tmp[prob_tmp==0] = 1 # ignore this term
    tmp = np.sum(-1*prob*np.log(prob_tmp),axis=-1)
    return tmp/np.log10(numclass)

def entropy_with_label(yinput, numclass=2):
    
    prob = np.zeros(numclass)
    samplecount = Counter(yinput[:,0])
    sampletot = yinput.shape[0]
    for i in range(numclass):
        
        prob[i] = samplecount[i]/sampletot
    
    return cal_entropy(prob, numclass = numclass)

def weighted_entropy_for_splitting(leftidx, rightidx, ynode, numclass):
    
    if leftidx[0].shape[0] == 0 or rightidx[0].shape[0] == 0:
        # invalid splitting
        return float('inf')
    
    lefty = ynode[leftidx]
    righty = ynode[rightidx]

    # use majority class as prediction target
    leftentropy = entropy_with_label(lefty, numclass = numclass)
    rightentropy = entropy_with_label(righty, numclass = numclass)
    
    weightedentropy = len(lefty)/len(ynode) * leftentropy + len(righty)/len(ynode) * rightentropy
    # print(len(lefty), len(righty))
    # print("left:", leftentropy, "right:", rightentropy, "weighted:", weightedentropy)
    return weightedentropy