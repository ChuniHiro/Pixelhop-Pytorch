#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 18:59:17 2021

@author: hongyu
"""


from skimage.util import view_as_windows
import numpy as np
import cv2
from lxml import etree
from scipy import optimize
from scipy import special
import numba
from sklearn.decomposition import PCA, IncrementalPCA


PASCAL_VOC_CLASSES = {'aeroplane':0, 
                      'bicycle':1, 
                      'bird':2, 
                      'boat':3,
                      'bottle':4, 
                      'bus':5, 
                      'car':6, 
                      'cat':7, 
                      'chair':8,
                      'cow':9, 
                      'diningtable':10, 
                      'dog':11, 
                      'horse':12,
                      'motorbike':13, 
                      'person':14, 
                      'pottedplant':15,
                      'sheep':16, 
                      'sofa':17,
                      'train':18, 
                      'tvmonitor':19
                      }
PASCAL_VOC_CLASSES_NAME = ('aeroplane', 
                      'bicycle', 
                      'bird', 
                      'boat',
                      'bottle', 
                      'bus', 
                      'car', 
                      'cat', 
                      'chair',
                      'cow', 
                      'diningtable', 
                      'dog', 
                      'horse',
                      'motorbike', 
                      'person', 
                      'pottedplant',
                      'sheep', 
                      'sofa', 
                      'train', 
                      'tvmonitor')

def Shrink(X, shrinkArg):
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    ch = X.shape[-1]
    X = view_as_windows(X, (1,win,win,ch), (1,stride,stride,ch))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1) # reshaped spatialdomainhere

def ShrinkXY(X, shrinkArg):
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    ch = X.shape[-1]
    X = view_as_windows(X, (1,win[0],win[1],ch), (1,stride[0],stride[1],ch))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1) # reshaped spatialdomainhere



def NoShrink(X, shrinkArg):

    return X

def ShrinkCross(X, shrinkArg):
    
    win = shrinkArg['win'] 
    stride = shrinkArg['stride']
    ch = X.shape[-1]
    # print("Shrink Cross Input", X.shape, "winsize=", win, "stride:", stride)
    X = view_as_windows(X, (1,win,win,ch), (1,stride,stride,ch))
    # print("Shrink Cross before selection", X.shape)
    # print("Shrink Cross 2", X.shape)
    X = X[:,:,:,0,0,[0,1,2,3,4,2,2,2,2],[2,2,2,2,2,0,1,3,4],:]
    
    # X = np.delete(X, [win//2], axis=-2) 
    # print("Shrink Cross Out:", X.shape)
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1) # reshaped spatialdomain here

def ShrinkDiamond(X, shrinkArg):
    
    win = shrinkArg['win'] 
    stride = shrinkArg['stride']
    ch = X.shape[-1]
    # print("Shrink Cross Input", X.shape, "winsize=", win, "stride:", stride)
    X = view_as_windows(X, (1,win,win,ch), (1,stride,stride,ch))
    # print("Shrink Cross before selection", X.shape)
    # print("Shrink Cross 2", X.shape)
    X = X[:,:,:,0,0,[0,1,1,2,2,2,3,3,2],[2,1,3,0,2,4,1,3,2],:]
    
    # X = np.delete(X, [win//2], axis=-2) 
    # print("Shrink Cross Out:", X.shape)
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1) # reshaped spatialdomain here


def ShrinkReverseCross(X, shrinkArg):
    
    win = shrinkArg['win'] 
    stride = shrinkArg['stride']
    ch = X.shape[-1]
    # print("Shrink Cross Input", X.shape, "winsize=", win, "stride:", stride)
    X = view_as_windows(X, (1,win,win,ch), (1,stride,stride,ch))
    # print("Shrink Cross before selection", X.shape)
    # print("Shrink Cross 2", X.shape)
    X = X[:,:,:,0,0,[0,1,2,3,4,0,1,3,4],[0,1,2,3,4,4,3,1,0],:]
    
    # X = np.delete(X, [win//2], axis=-2) 
    # print("Shrink Cross Out:", X.shape)
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1) # reshaped spatialdomain here


def Concat(X, concatArg):
    return X

def CoordinateConvert(Sto, Sfrom, pt):
    # dimension 0 is x
    # dimension 1 is y
    # midn dimensions can be quite different in different cases
    scalex = Sto[1]/Sfrom[1]
    scaley = Sto[0]/Sfrom[0]
    xori, yori = pt
    xnew, ynew = int(xori*scalex), int(yori*scaley)
    return (xnew, ynew)

def coloraug(original_image):
    
    renorm_image = np.reshape(original_image,(original_image.shape[0]*original_image.shape[1],3))
    renorm_image = renorm_image.astype('float32')
    mean = np.mean(renorm_image, axis=0)
    std = np.std(renorm_image, axis=0)
    #normalize
    renorm_image -= mean
    renorm_image /= std
    
    renorm_image_plot = renorm_image.reshape(original_image.shape[0], original_image.shape[1], -1)
    # svd
    cov = np.cov(renorm_image, rowvar=False) #covariance matrix
    lambdas, p = np.linalg.eig(cov) # eigenvector and eigenvalue
    alphas = np.random.normal(0, 0.1, 3) #random weights
    
    delta = np.dot(p, alphas*lambdas) # eigenvector * (alpha * lambda)T

    pca_augmentation_version_renorm_image = renorm_image + delta
    #reconstruct
    pca_color_image = pca_augmentation_version_renorm_image * std + mean
    pca_color_image = np.maximum(np.minimum(pca_color_image, 255), 0).astype('uint8')
    return pca_color_image.reshape(original_image.shape[0], original_image.shape[1], -1)

def remove_mean_axis(X, axis):
    feature_mean = np.mean(X, axis=axis , keepdims=True)
    X = X - feature_mean
    return X

def showsinglebbox(img, bbox):
    
    imbox = img.copy();
    
    xbox, ybox, wbox, hbox = bbox[:4]
    xbox, ybox, wbox, hbox = int(xbox), int(ybox), int(wbox),int(hbox)
    cv2.rectangle(imbox, (xbox, ybox), (xbox + wbox, ybox + hbox), (0, 0 , 255), 2, cv2.LINE_AA)
    # cv2.putText(imbox,str(bbox[1]), (xbox,ybox), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    
    return imbox

def showbbox_wh(img, bboxes, color=(0, 0 , 255), linewidth = 1):
    
    imbox = img.copy();
    
    for bbox in bboxes:
        # print(bbox)
        if len(bbox) == 2:
            bbox = bbox[0]
            
        xbox, ybox, wbox, hbox = bbox[:4]
        # xbox = max(1, xbox)
        # ybox = max(1, ybox)
        xbox, ybox, wbox, hbox = int(xbox), int(ybox), int(wbox),int(hbox)
        cv2.rectangle(imbox, (xbox, ybox), (xbox + wbox, ybox + hbox), color, linewidth)
        
        # image = cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA)
    return imbox

def showbbox_coordinate(img, bboxes, color=(0, 255 , 0)):
    
    imbox = img.copy();
    
    for bbox in bboxes:
        # print(bbox)
        if len(bbox) == 2:
            bbox = bbox[0]
            
        xbox, ybox, xbox2, ybox2 = bbox[:4]
        xbox, ybox, xbox2, ybox2  = int(xbox), int(ybox), int(xbox2),int(ybox2)
        # print(xbox,ybox,xbox2,ybox2)
        cv2.rectangle(imbox, (xbox, ybox), (xbox2, ybox2), color, 1)
        # plt.imshow(imbox)
        # plt.show()
        # image = cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA)
    return imbox


def showbboxwithText(img, bboxes, classlabel, color=(0, 0 , 255)):
    
    imbox = img.copy();
    
    for idx,bbox in enumerate(bboxes):

        if len(bbox) == 2:
            
            classlabel = bbox[1]
            bbox = bbox[0]
        
        else:
            classlabel = bbox[-1]
            
        xbox, ybox, wbox, hbox = bbox[:4]
        xbox, ybox, wbox, hbox = int(xbox), int(ybox), int(wbox),int(hbox)
        cv2.rectangle(imbox, (xbox, ybox), (xbox + wbox, ybox + hbox), color, 2, cv2.LINE_AA)
        cv2.putText(imbox,PASCAL_VOC_CLASSES_NAME[classlabel], (xbox,ybox), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
        
    return imbox

def neighborhoodpooling(point, featmap, winsize = 3):
    
    if winsize == 0 :
        return featmap[point]
    
    xx, yy = point
    featmap = np.pad(featmap, ((winsize, winsize ), (winsize , winsize), (0,0)),'reflect')
    xx += winsize
    yy += winsize
    
    featdim = featmap.shape[-1]
    featpoint = np.zeros(featdim)
    for dim in range(featdim):
        
        featmax = abs(featmap[xx-winsize: xx + winsize + 1 , yy - winsize: yy + winsize +1,dim]).max()
        featmax *= np.sign(featmap[xx-winsize: xx + winsize + 1 , yy - winsize: yy + winsize +1,dim].max())
        featpoint[dim] = featmax
    
    return featpoint
    
class PascalVOCXML:
    
    def __init__(self, xml_pth):
        self.tree = etree.parse(xml_pth)
        self.boxes = []


    def get_boxes_2007(self):
        if len(self.boxes) == 0:
            for obj in self.tree.xpath('//object'):

                for item in obj.getchildren():
                    if (item.tag=='name'):
                        classname = item.text
                        classid = PASCAL_VOC_CLASSES[classname]
                        
                    if (item.tag=='bndbox'):
                        coords = [int(float(_.text)) for _ in item.getchildren()]
                        x1, y1, x2, y2 = coords
                
                # if record class label here
                self.boxes.append([[x1, y1, x2-x1, y2-y1], classid])
                # self.boxes.append([x1, y1, x2-x1, y2-y1])
        return self.boxes
    
    def get_boxes_2012(self):
        if len(self.boxes) == 0:
            for obj in self.tree.xpath('//object'):

                for item in obj.getchildren():
                    if (item.tag=='name'):
                        classname = item.text
                        classid = PASCAL_VOC_CLASSES[classname]
                        
                    if (item.tag=='bndbox'):
                        coords = [int(float(_.text)) for _ in item.getchildren()]
                        x2, x1, y2, y1 = coords
                
                # if record class label here
                self.boxes.append([[x1, y1, x2-x1, y2-y1], classid])
                # self.boxes.append([x1, y1, x2-x1, y2-y1])
        return self.boxes
"""
module 2
"""


def in_gt_boxes(coordinate, gtboxes, inputresolution, imgresolution):
    """
    Applied to bounding boxes with:
    gtboxes- x, y , w , h
    coordinate- point-y,x
    """
    scale = imgresolution//inputresolution
    ytmp, xtmp = coordinate
    centerx, centery = (xtmp * scale + (xtmp+1) * scale - 1)/2, (ytmp * scale + (ytmp+1) * scale - 1) / 2
    # print('check center:', centerx,centery)
    for bbox in gtboxes:
        # print(bbox)
        xbox, ybox, wbox, hbox = bbox
        if xbox <= centerx <= xbox + wbox and ybox <= centery <= ybox + hbox:
        # if xbox < centerx < xbox + wbox and ybox < centery < ybox + hbox:
            # print(xbox,ybox,wbox,hbox)
            return True
        
        # get a solfter version?? 
        # respect to RF
        
    return False

def in_gt_boxes_scale(coordinate, gtboxes, inputresolution, imgresolution, receptivefield = None):
    """
    Applied to bounding boxes with:
    gtboxes- x, y , w , h
    coordinate- point-y,x
    """
    scale = imgresolution//inputresolution
    scaleth = 3
    ytmp, xtmp = coordinate
    centerx, centery = (xtmp * scale + (xtmp+1) * scale - 1)/2, (ytmp * scale + (ytmp+1) * scale - 1) / 2
    # print('check center:', centerx,centery)
    for bbox in gtboxes:
        # print(bbox)
        xbox, ybox, wbox, hbox = bbox
        if (wbox < receptivefield/scaleth or wbox > scaleth * receptivefield) and \
            (hbox < receptivefield/scaleth or hbox > scaleth * receptivefield): 
            
            # print("object scale not matched!", wbox, hbox, receptivefield)
            continue
        # print(bbox)
        if xbox <= centerx <= xbox + wbox and ybox <= centery <= ybox + hbox:
        # if xbox < centerx < xbox + wbox and ybox < centery < ybox + hbox:
            # print(xbox,ybox,wbox,hbox)
            return True
        
        # get a solfter version?? 
        # respect to RF
        
    return False


def in_gt_boxes(coordinate, gtboxes, inputresolution, imgresolution, receptivefield = None):
    """
    Applied to bounding boxes with:
    gtboxes- x, y , w , h
    coordinate- point-y,x
    """
    scale = imgresolution//inputresolution
    scaleth = 3
    ytmp, xtmp = coordinate
    centerx, centery = (xtmp * scale + (xtmp+1) * scale - 1)/2, (ytmp * scale + (ytmp+1) * scale - 1) / 2
    # print('check center:', centerx,centery)
    for bbox in gtboxes:
        # print(bbox)
        xbox, ybox, wbox, hbox = bbox[:4]
        if (wbox < receptivefield/scaleth or wbox > scaleth * receptivefield) and \
            (hbox < receptivefield/scaleth or hbox > scaleth * receptivefield): 
            
            # print("object scale not matched!", wbox, hbox, receptivefield)
            continue
        # print(bbox)
        if xbox <= centerx <= xbox + wbox and ybox <= centery <= ybox + hbox:
        # if xbox < centerx < xbox + wbox and ybox < centery < ybox + hbox:
            # print(xbox,ybox,wbox,hbox)
            return True
        
        # get a solfter version?? 
        # respect to RF
        
    return False


def collectxgbsample(Xsaab, gtboxes, imgresolution=112, datatype = 'float32', RF=None):
    
    positive_samples= []
    negative_samples= []
        
    Ssaab = Xsaab.shape
    featresolution = Ssaab[0]
    for xx in range(featresolution):
        for yy in range(featresolution):
            
            if in_gt_boxes((xx,yy), gtboxes, featresolution,imgresolution):
                
                positive_samples.append(Xsaab[xx][yy].astype(datatype))
                
            else:
                
                negative_samples.append(Xsaab[xx][yy].astype(datatype))
    # print(len(positive_samples), len(negative_samples))
    return positive_samples, negative_samples
                    
def xgbpred(Xsaab, xgb):
    
    Ssaab = Xsaab.shape
    featresolution = Ssaab[0]
    predmap = np.zeros((featresolution, featresolution))
    # predbinary = np.zeros((featresolution, featresolution))
    
    # NO FOR LOOP PLZ!
    # COLLECT ALL FEATURE TOGETHER AND PREDICT ONCE!
    for xx in range(featresolution):
        for yy in range(featresolution):
    
            testpoint = Xsaab[xx][yy]
            testpoint = testpoint.reshape(-1,testpoint.shape[0])
            predtmp = xgb.predict_proba(testpoint)
            predmap[xx][yy] = predtmp[0][1]
            
            # quantization to [0,0.5,1]
            # if predtmp[0][1]< 0.25:
            #     predmap[xx][yy] = 0
            # elif 0.25 <= predtmp[0][1] <= 0.75:
            #     predmap[xx][yy] = 0.5
            # else:
            #     predmap[xx][yy] = 1
            
   
    # predmap = (predmap - predmap.min())/(predmap.max()- predmap.min())
    return predmap

def IOU(boxA, boxB):
    
    # For boxes with input as:
    # xbox, ybox, wbox, hbox
    # boxA[2] = boxA[2] + boxA[0]
    # boxB[2] = boxB[2] + boxB[0]
    
    # boxA[3] = boxA[3] + boxA[1]
    # boxB[3] = boxB[3] + boxB[1]
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # # v1 for weired integers?
    # interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # boxAArea = (boxA[2] - boxA[0] - 1) * (boxA[3] - boxA[1] - 1)
    # boxBArea = (boxB[2] - boxB[0] - 1) * (boxB[3] - boxB[1] - 1)

    # # v2 for float
    interArea = max(0, xB - xA ) * max(0, yB - yA )
    
    boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1] )
    boxBArea = (boxB[2] - boxB[0] ) * (boxB[3] - boxB[1] )

    iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
    return iou  

# def IOUv2(boxA, boxB):
    
#     # For boxes with input as:
#     # xbox, ybox, wbox, hbox
#     boxA[2] = boxA[2] + boxA[0]
#     boxB[2] = boxB[2] + boxB[0]
    
#     boxA[3] = boxA[3] + boxA[1]
#     boxB[3] = boxB[3] + boxB[1]
    
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
        
#     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

#     boxAArea = (boxA[2] - boxA[0] - 1) * (boxA[3] - boxA[1] - 1)
#     boxBArea = (boxB[2] - boxB[0] - 1) * (boxB[3] - boxB[1] - 1)

#     iou = interArea / float(boxAArea + boxBArea - interArea)
# 	# return the intersection over union value
#     return iou  


class FocalLoss:

    def __init__(self, gamma, alpha=None):
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(0, 1),
            method='bounded'
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds
        
    def lgb_obj(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        is_higher_better = False
        return 'focal_loss', self(y, p).mean(), is_higher_better

def logloss_init_score(y):
    """
    logloss initialization added!
    
    stop very early -- nice!

    """
    p = y.mean()
    p = np.clip(p, 1e-15, 1 - 1e-15)  # never hurts
    log_odds = np.log(p / (1 - p))
    return log_odds

def check_gradient(func, grad, values, eps=1e-8):
    
    approx = (func(values + eps) - func(values - eps)) / (2 * eps)
    return np.linalg.norm(approx - grad(values))

@numba.jit(nopython = True, parallel = True)
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

@numba.jit(forceobj = True, parallel = True)
def remove_mean(X: np.ndarray, feature_mean: np.ndarray):
    return X - feature_mean

@numba.jit(nopython=True)
# @numba.jit(nopython=True, parallel = True)
def feat_transform(X: np.ndarray, kernel: np.ndarray):
    return X @ kernel.transpose()


class Saabv2():
    """
    inputs are concatented pixelhop features
    no AC/DC split
    remove global mean0
    """
    
    def __init__(self, num_kernels=-1):
        
        self.Mean0 = []
        self.Energy = []
        self.num_kernels = num_kernels
        self.trained = False
    
    def fit(self, X):
        
        if self.num_kernels == -1:
            self.num_kernels = X.shape[-1]
        
        self.Mean0 = np.mean(X, axis = 0, keepdims = True)
        X = remove_mean(X, self.Mean0)
        
        pca = PCA(n_components=self.num_kernels, svd_solver = 'full' ).fit(X)
        kernels = pca.components_
        energy = pca.explained_variance_ / np.sum(pca.explained_variance_)
        
        self.Kernels, self.Energy = kernels.astype('float32'), energy
        self.trained = True
        
    def transform(self, X):
        
        X = remove_mean(X, self.Mean0)
        X = np.matmul(X, self.Kernels.transpose()).astype('float32')
        
        return X
        
        
class cwSpatialPCA():

    def __init__ (self, mymodel_enegy):
        self.saab_all = []
        self.prevEnergy = mymodel_enegy 
        self.Energy = []

    def cwSaab(self, X, train = False):
        
        # x : [samples, features
        # train: flag for train or not

        if train == True:
            saab_cur = []
        else:
            saab_cur = self.saab_all

        transformed = []
        S = list(X.shape) # sample,W,H,channel
        S[-1] = 1
        X = np.moveaxis(X, -1, 0)
        
        leni = X.shape[0]
#         print("X:", X.shape)
#         print("check input energy",self.prevEnergy.shape)
    
        for i in range(leni): # for each channel
            X_tmp = X[i].reshape(S)
            if train == True:
                saab = self.SaabFit(X_tmp)
                self.saab_all.append(saab)
                # bias_cur.append(saab.Bias_current)#*np.ones(saab.Energy.size))
#                 eng.append(saab.Energy * self.prevEnergy[i])
                self.Energy.append(saab.Energy * self.prevEnergy[i])
                transformed.append(self.SaabTransform(X_tmp, saab=saab))
            else:
                if len(saab_cur) == i:
                    break
                transformed.append(self.SaabTransform(X_tmp, saab=saab_cur[i]))
                
        transformed = np.concatenate(transformed, axis=-1)
#         print("transformed:", transformed.shape)
#         eng = np.concatenate(eng)
        if train == True:
            self.Energy = np.concatenate(self.Energy)
        return transformed

    def SaabFit(self, X):
        
#         print("fit",X.shape)
        S = list(X.shape)
        X = X.reshape(S[0], -1)
#         print("fit2",X.shape)
        saab = Saabv2(num_kernels=-1)
        saab.fit(X)
        return saab

    def SaabTransform(self, X, saab):

#         S = list(X.shape)
        X = X.reshape(X.shape[0], -1)
#         print("SaabTransform,X",  X.shape)
        transformed = saab.transform(X)
#         print("SaabTransform2,X",  X.shape)
#         transformed = transformed.reshape(S[0],S[1],S[2],-1)
#         print("SaabTransform3,X",  X.shape)
        return transformed
    
def nms(bounding_boxes, confidence_score, threshold):
    
    """
        Non-max Suppression Algorithm
        @param list  Object candidate bounding boxes
        @param list  Confidence score of bounding boxes
        @param float IoU threshold
        @return Rest boxes after nms operation
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)
    # print(boxes)
    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score

