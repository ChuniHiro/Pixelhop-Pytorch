'''
v2021.05.31

10cls baseline, train+test
label smoothing
decision (mid) fusion for single color only

'''

import pickle
from keras.datasets import cifar10
import numpy as np
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import RandomForestClassifier as RFC
# from sklearn.ensemble import RandomForestRegressor as RFR
# from sklearn.metrics import log_loss as LL
# from sklearn.cluster import KMeans,MiniBatchKMeans
# import numpy.linalg as LA
# import math
# from sklearn.decomposition import PCA
import xgboost as xgb
from tensorflow.python.platform import flags
# from sklearn.svm import SVC
import time
import os
import cv2
# import gc
# import lib.data as data
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,MiniBatchKMeans
from skimage.measure import block_reduce
from skimage.util import view_as_windows
# from skimage.measure import block_reduce
# from skimage.util import view_as_windows
#%% self
# import lib.lib_competitions_1020 as COMPETE
# import lib.utils as utils
import lib.lib_tuning as TUNE
import lib.feat_utils as FEAT
# import lib.lib_attention as ATT
# import lib.lib_spatialPCA_EEC_img as SPCA
import lib.lib_stats as STAT

import lib.layer as layer


import lib.lib_iterative_graph_train_CP0524_10cls as Iter_Graph_Train_CP
import numba

# import lib.lib_plot as PLT_LIB
# import lib.lib_data as DATA
# from lib.lib_LAC import MultiModelXGB, LAC
# import lib.lib_LAC as CLUSTER


#%% warning
import warnings
warnings.filterwarnings("ignore")
#%%
# flags.DEFINE_string("root1", '/media/bingbing/MyData/ICPR_CIFAR/baseline_result/cwsaab_cifar10_0125_10cls_3x3_6layers_PQR/', "root")
# flags.DEFINE_string("root2", '/media/bingbing/MyData/ICPR_CIFAR/baseline_result/cwsaab_cifar10_0109_10cls_3x3_eecQ/', "root")
# flags.DEFINE_string("root3", '/media/bingbing/MyData/ICPR_CIFAR/baseline_result/cwsaab_cifar10_0109_10cls_3x3_eecR/', "root")

# flags.DEFINE_string("root", '/media/bingbing/MyData/ICPR_CIFAR/baseline_result/cwsaab_cifar_0515_AUG8/cwsaab_cifar_0515_AUG8', "root")
flags.DEFINE_string("root", '/media/hongyu/SSD/SSDUBUNTU/WUSL/Pixelhop-Pytorch/Epixelhop/save', "root")
# flags.DEFINE_string("root", '/mnt/yijing/cifar/cwsaab_cifar10_0310_3x3_AUG_35_img_', "root")

flags.DEFINE_string("subroot", 'P', "root")
flags.DEFINE_string("HOPLIST","1,2,3", "augmentation mode")# LAG

# soft label smoothing hyperparameters
flags.DEFINE_integer("agg", 9, "number of aggregation window 9=3x3, 25=5x5")# LAG
flags.DEFINE_integer("rank_neighbour",0, "rankd the siblings/children")# LAG
flags.DEFINE_integer("incre_train", 0, "incrementally train XGBoost or not")# LAG
flags.DEFINE_string("eval", 'mlogloss', "loss function for evaluation")# LAG
flags.DEFINE_integer("saabfeat", 0, "use Saab features after 2nd iteration or not")# LAG
flags.DEFINE_integer("average", 0, "average pooling the probability")# LAG
flags.DEFINE_integer("parent", 1, "consider parent or not")# LAG
flags.DEFINE_integer("version", 5, "neighborhood version")# version 1234
flags.DEFINE_integer("neigh4", 0, "0: 8neighbor, 1: 4neighbor")# version 1234
flags.DEFINE_integer("NODC",1, "discard DC or not")# LAG
flags.DEFINE_integer("MAXITER",3, "max iteration number")# LAG
flags.DEFINE_integer("small",1, "xgboost classifier model size")# LAG


flags.DEFINE_integer("pair0",3, "augmentation mode")# LAG
flags.DEFINE_integer("pair1",5, "augmentation mode")# LAG
flags.DEFINE_string("GPU",None, "augmentation mode")# LAG
flags.DEFINE_string("AUG","0", "augmentation mode")# LAG
flags.DEFINE_integer("POOL",3, "COMPACT ENERGY OR NOT")
# flags.DEFINE_integer("HOP",2, "Hop number (start from 0)")
flags.DEFINE_integer("ENERGY",0, "COMPACT ENERGY OR NOT")
flags.DEFINE_integer("SPCA",0, "COMPACT ENERGY OR NOT")
flags.DEFINE_float("SELECT",0.7, "dimension of feature selection")# LAG
flags.DEFINE_integer("VALID",0, "dimension of feature selection")# LAG
# flags.DEFINE_string("saveroot", '/media/bingbing/MyData/ICPR_CIFAR/VCIP2021_cifar/baseline_0524_aug8_v1', "root")
flags.DEFINE_string("saveroot", '/media/hongyu/SSD/SSDUBUNTU/WUSL/Pixelhop-Pytorch/Epixelhop/save/baseline', "root")
flags.DEFINE_string("imgtype",'PQR', "image type")# LAG
flags.DEFINE_string("FStype",'ANOVA', "Feature selection mode: CE, ANOVA, XGB")# LAG

FLAGS = flags.FLAGS

FStype = FLAGS.FStype
if FLAGS.subroot =='PQR':
    root = [FLAGS.root+'P/',FLAGS.root+'Q/',FLAGS.root+'PQR_R/']
else:
    # root = [FLAGS.root + FLAGS.subroot +'/']
    root = [FLAGS.root + '/']
saveroot = FLAGS.saveroot+'/'  

if not os.path.isdir(saveroot):os.makedirs(saveroot)

AUG_list = [int(s) for s in FLAGS.AUG.split(',')]
HOPLIST = [int(s) for s in FLAGS.HOPLIST.split(',')]
sub = '_DC{}_agg{}_r{}_s{}_a{}_p{}_4nei{}_small{}_max{}_version{}'.format(1-FLAGS.NODC,FLAGS.agg,FLAGS.rank_neighbour,FLAGS.saabfeat,FLAGS.average, FLAGS.parent, FLAGS.neigh4, FLAGS.small, FLAGS.MAXITER, FLAGS.version)

with open(saveroot + 'sig{}.npy'.format(sub), 'wb') as f:
    np.save(f, np.array([]))


BS = 2000


@numba.jit(forceobj = True, parallel = True)
def remove_std(X: np.ndarray, STD: np.ndarray):
    return X/STD

@numba.jit(forceobj = True, parallel = True)
def cal_std(X: np.ndarray):
    return np.std(X, axis=0, keepdims=1).reshape(1,1,1,-1)

def get_feat_grouping_raw(hopidx=0,target=[3,5],mode='tr',ROOT = None, chosenidx=None, aug_idx=0):
    if chosenidx is None:
        if mode=='tr':
            with open(ROOT+'tr_aug'+str(aug_idx)+'_feature_Hop'+str(hopidx+1)+'_all_AUG8.npy', 'rb') as f:
                X_train_all = np.load(f)[:,:,:,FLAGS.NODC:]
            return X_train_all
        else:
            with open(ROOT+'te_aug'+str(aug_idx)+'_feature_Hop'+str(hopidx+1)+'_all_AUG8.npy', 'rb') as f:
                X_test_all = np.load(f)[:,:,:,FLAGS.NODC:]
            return X_test_all
    else:
        if mode=='tr':
            with open(ROOT+'tr_aug'+str(aug_idx)+'_feature_Hop'+str(hopidx+1)+'_all_AUG8.npy', 'rb') as f:
                X_train_all = np.load(f)[:,:,:,FLAGS.NODC:][chosenidx]
            return X_train_all
        else:
            with open(ROOT+'te_aug'+str(aug_idx)+'_feature_Hop'+str(hopidx+1)+'_all_AUG8.npy', 'rb') as f:
                X_test_all = np.load(f)[:,:,:,FLAGS.NODC:][chosenidx]
            return X_test_all



# def load_feat_allAUG_2(AUG_list, HOPLIST, tr_chosen, te_chosen):
#     tr_X_all, te_X_all = [],[]
    
#     for hop in HOPLIST:
#         tr_X = []
#         for aug in AUG_list:
#             tr_X.append(get_feat_grouping_raw(hopidx = hop, mode='tr', ROOT = root[0], aug_idx=aug, chosenidx=tr_chosen[:50000]))
#         tr_X = np.concatenate(tr_X, axis=0)
        
#         te_X = []
#         for aug in AUG_list:
#             te_X.append(get_feat_grouping_raw(hopidx = hop, mode='te', ROOT = root[0], aug_idx=aug, chosenidx=te_chosen[:10000]))
#         te_X = np.concatenate(te_X, axis=0)

#         ch = tr_X.shape[-1]
#         if hop < 2: #
#             if FLAGS.POOL==2:
#                 tr_X = block_reduce(tr_X, (1, FLAGS.POOL, FLAGS.POOL, 1), np.max)
#                 te_X = block_reduce(te_X, (1, FLAGS.POOL, FLAGS.POOL, 1), np.max)
#             elif FLAGS.POOL==3: # overlapping pool
#                 tr_X = view_as_windows(tr_X, (1,FLAGS.POOL,FLAGS.POOL,ch), (1,2,2,ch))
#                 tr_X = tr_X.reshape(tr_X.shape[0],tr_X.shape[1],tr_X.shape[2],FLAGS.POOL*FLAGS.POOL,-1)
#                 tr_X = np.max(tr_X,axis=3)
#                 te_X = view_as_windows(te_X, (1,FLAGS.POOL,FLAGS.POOL,ch), (1,2,2,ch))
#                 te_X = te_X.reshape(te_X.shape[0],te_X.shape[1],te_X.shape[2],FLAGS.POOL*FLAGS.POOL,-1)
#                 te_X = np.max(te_X,axis=3)
                
#         tr_X_all.append(tr_X)
#         te_X_all.append(te_X)
    
#     return tr_X_all, te_X_all

# def load_feat_allAUG(AUG_list, HOPLIST):
#     tr_X_all, te_X_all = [],[]
    
#     for hop in HOPLIST:
#         tr_X = []
#         for aug in AUG_list:
#             tr_X.append(get_feat_grouping_raw(hopidx = hop, mode='tr', ROOT = root[0], aug_idx=aug))#, chosenidx=tr_chosen)
#         tr_X = np.concatenate(tr_X, axis=0)
        
#         te_X = []
#         for aug in AUG_list:
#             te_X.append(get_feat_grouping_raw(hopidx = hop, mode='te', ROOT = root[0], aug_idx=aug))#, chosenidx=tr_chosen)
#         te_X = np.concatenate(te_X, axis=0)

#         ch = tr_X.shape[-1]
#         if hop < 2: #
#             if FLAGS.POOL==2:
#                 tr_X = block_reduce(tr_X, (1, FLAGS.POOL, FLAGS.POOL, 1), np.max)
#                 te_X = block_reduce(te_X, (1, FLAGS.POOL, FLAGS.POOL, 1), np.max)
#             elif FLAGS.POOL==3: # overlapping pool
#                 tr_X = view_as_windows(tr_X, (1,FLAGS.POOL,FLAGS.POOL,ch), (1,2,2,ch))
#                 tr_X = tr_X.reshape(tr_X.shape[0],tr_X.shape[1],tr_X.shape[2],FLAGS.POOL*FLAGS.POOL,-1)
#                 tr_X = np.max(tr_X,axis=3)
#                 te_X = view_as_windows(te_X, (1,FLAGS.POOL,FLAGS.POOL,ch), (1,2,2,ch))
#                 te_X = te_X.reshape(te_X.shape[0],te_X.shape[1],te_X.shape[2],FLAGS.POOL*FLAGS.POOL,-1)
#                 te_X = np.max(te_X,axis=3)
                
#         tr_X_all.append(tr_X)
#         te_X_all.append(te_X)
    
#     return tr_X_all, te_X_all


def load_feat_allAUG_singleHop(AUG_list, mode='tr', hop=0):
    tr_X = []
    for aug in AUG_list:
        print('aug{}'.format(aug))
        tr_X.append(get_feat_grouping_raw(hopidx = hop, mode=mode, ROOT = root[0], aug_idx=aug))#, chosenidx=tr_chosen)
    tr_X = np.concatenate(tr_X, axis=0)

    ch = tr_X.shape[-1]
    if hop < 2: #
        if FLAGS.POOL==2:
            tr_X = block_reduce(tr_X, (1, FLAGS.POOL, FLAGS.POOL, 1), np.max)
        elif FLAGS.POOL==3: # overlapping pool
            tr_X = view_as_windows(tr_X, (1,FLAGS.POOL,FLAGS.POOL,ch), (1,2,2,ch))
            tr_X = tr_X.reshape(tr_X.shape[0],tr_X.shape[1],tr_X.shape[2],FLAGS.POOL*FLAGS.POOL,-1)
            tr_X = np.max(tr_X,axis=3)
    
    return tr_X



def logical_AND_multi(log1,log2,*args):
    length = len(args)
    indicator = log1*log2
    for i in range(length):
        indicator*=args[i]
    return indicator

def logical_OR_multi(log1,log2,*args):
    length = len(args)
    indicator = log1+log2
    for i in range(length):
        indicator+=args[i]
    return indicator


# def get_groups_TRAIN(tr_X, tr_y, PAIR0, PAIR1):
#     tr_chosen = logical_OR_multi(tr_y == PAIR0, tr_y == PAIR1).squeeze()
#     ynew = np.copy(tr_y[tr_chosen])
#     ynew[ynew==PAIR0] = 0
#     ynew[ynew==PAIR1] = 1
    
#     tr_X_new = []
#     for i in range(len(tr_X)):
#         tr_X_new.append(np.copy(tr_X[i][tr_chosen]))
    
#     return tr_X_new, ynew, tr_chosen
    
# def get_groups_TEST(X, y, y_top2, PAIR0, PAIR1):
#     '''top2 in pair0 pair1, and ground truth also pair0 pair1'''
#     chosen = logical_AND_multi(logical_AND_multi(y_top2[:,0] == PAIR0, y_top2[:,1] == PAIR1).squeeze(), logical_OR_multi(y == PAIR0, y == PAIR1).squeeze())
#     ynew = np.copy(y[chosen])
#     ynew[ynew==PAIR0] = 0
#     ynew[ynew==PAIR1] = 1
    
#     X_new = []
#     for i in range(len(X)):
#         X_new.append(np.copy(X[i][chosen]))
        
#     print('test for {} vs {} = {}'.format(PAIR0, PAIR1, np.sum(chosen)))
#     return X_new, ynew, chosen

def get_groups_TRAIN_2(tr_y, PAIR0, PAIR1):
    tr_chosen = logical_OR_multi(tr_y == PAIR0, tr_y == PAIR1).squeeze()
    ynew = np.copy(tr_y[tr_chosen])
    ynew[ynew==PAIR0] = 0
    ynew[ynew==PAIR1] = 1
    print(np.unique(ynew))
    
    return ynew, tr_chosen
    
def get_groups_TEST_2(y, y_top2, PAIR0, PAIR1):
    '''top2 in pair0 pair1, and ground truth also pair0 pair1'''
    y_top2 = np.tile(y_top2.T, len(AUG_list)).T
    chosen = logical_AND_multi(logical_AND_multi(y_top2[:,0] == PAIR0, y_top2[:,1] == PAIR1).squeeze(), logical_OR_multi(y == PAIR0, y == PAIR1).squeeze())
    ynew = np.copy(y[chosen])
    ynew[ynew==PAIR0] = 0
    ynew[ynew==PAIR1] = 1
    print(np.unique(ynew))
    print('test for {} vs {} = {}'.format(PAIR0, PAIR1, np.sum(chosen)))
    return ynew, chosen


    
def cal_hist(prob, bins=None):
    prob_hist = np.zeros((prob.shape[0],bins.size-1))
    for i in range(bins.size-1):
        logic = (prob>=bins[i])*(prob<bins[i+1])
        prob_hist[:,i] = np.sum(logic>0, axis=1)
    return prob_hist

def cal_top2_acc(y_prob, y_gt, top=2):
    te_argsort = np.argsort(-1*y_prob,axis=1)
    
    # top2 acc
    te_top2_pred = te_argsort[:,:top]
    tmp = te_top2_pred - y_gt.reshape(-1,1)
    te_top2_acc = np.sum(np.min(np.abs(tmp),axis=1)==0)/y_gt.size
    
    return te_top2_acc



if __name__=='__main__':
 	# read data
    (train_images, y_train), (test_images, y_test) = cifar10.load_data()
    train_images = train_images/255.0   
    test_images = test_images/255.0  
    
    y_train = y_train.squeeze().astype('int')
    y_test = y_test.squeeze().astype('int')
    
    NUM_CLS = 10
    # HOPLIST = [1,2,3] 
    
    # tr_feat_allHop, te_feat_allHop = load_feat_allAUG(AUG_list, HOPLIST)
    
    if len(AUG_list)>1:    
        # train_images = layer.augment_combine(train_images,mode=AUG_list)
        # train_images = np.concatenate(train_images,axis=0)
        y_train = np.tile(y_train, len(AUG_list))
        y_test = np.tile(y_test, len(AUG_list))

 #%%   
    START_TIME = time.time()
    # tr_acc_pairwise = np.zeros((10,10))
    # te_acc_pairwise = np.zeros((10,10))
    te_acc_updated_all = []
    # te_acc_updated_all.append(np.sum(np.argmax(te_prob_M2A,axis=1)==y_test[:10000].reshape(-1))/10000.)

    # tr_X_ss_list, te_X_ss_list = load_feat_allAUG(AUG_list, HOPLIST)

    y_train_pix_list, y_test_pix_list = [],[]
    tr_X_ss_list, te_X_ss_list = [],[]

    selected_idx_all = []
    for i in range(len(HOPLIST)):
        print(i)
        tr_X_ss = load_feat_allAUG_singleHop(AUG_list, mode='tr', hop=HOPLIST[i])
        tr_NN, HH, WW, CC = tr_X_ss.shape
        # STD = np.std(tr_X_ss.reshape(-1,CC), axis=0, keepdims=1).reshape(1,1,1,-1)
        # tr_X_ss = tr_X_ss/STD
        print('std')
        STD = cal_std(tr_X_ss.reshape(-1,CC))
        tr_X_ss = remove_std(tr_X_ss,STD)
        y_train_pix_list.append(np.repeat(y_train, HH*WW))
        y_test_pix_list.append(np.repeat(y_test, HH*WW))

        #% Feature selection
        print('feature select')
        # tr_X_ss, te_X_ss, selected_idx,_ = FEAT.feature_selection(tr_X_ss.reshape(-1,CC), y_train_pix_list[i], te_X_ss.reshape(-1,CC), FStype= 'ANOVA',thrs=FLAGS.SELECT)  
        tr_X_ss, selected_idx,_ = FEAT.feature_selection_train(tr_X_ss.reshape(-1,CC), y_train_pix_list[i],FStype= 'ANOVA',thrs=FLAGS.SELECT)  
        selected_idx_all.append(selected_idx)
        tr_X_ss_list.append(tr_X_ss.reshape(tr_NN, HH, WW, -1))
        
        print('test FS')
        te_X_ss = load_feat_allAUG_singleHop(AUG_list, mode='te', hop=HOPLIST[i])
        te_NN, _,_,_ = te_X_ss.shape
        te_X_ss = te_X_ss[:,:,:,selected_idx]
        # te_X_ss = te_X_ss/(STD[:,:,:,selected_idx])
        te_X_ss = remove_std(te_X_ss,STD[:,:,:,selected_idx])
        te_X_ss_list.append(te_X_ss)

        
    fwrite = open(saveroot+'selected_idx_all.pkl','wb')
    pickle.dump(selected_idx_all, fwrite,protocol=2)
    fwrite.close()

    START_TIME = time.time()
    print('start Soft label smoothings')
    clf = Iter_Graph_Train_CP.iter_clf(max_iter=FLAGS.MAXITER, agg=FLAGS.agg, rank_neighbour=FLAGS.rank_neighbour, 
                              incre_train = FLAGS.incre_train, eval_metric=FLAGS.eval,
                              gpu = FLAGS.GPU,
                              saabfeat = FLAGS.saabfeat,
                              average = FLAGS.average,
                              parent = FLAGS.parent,
                              version=FLAGS.version,
                              small_model=FLAGS.small,
                              neigh4=FLAGS.neigh4)
    clf.fit(tr_X_ss_list, y_train_pix_list, te_X_ss_list, y_test_pix_list)
    
    fwrite = open(saveroot+'tr_prob_list.pkl','wb')
    pickle.dump(clf.tr_prob_save, fwrite,protocol=2)
    fwrite.close()
    
    fwrite = open(saveroot+'te_prob_list.pkl','wb')
    pickle.dump(clf.te_prob_save, fwrite,protocol=2)
    fwrite.close()
    

    fwrite = open(saveroot+'modelCP.pkl','wb')
    pickle.dump(clf.models, fwrite, protocol=2)
    fwrite.close()
    
    for hopidx in range(len(HOPLIST)):
        for iteidx in range(FLAGS.MAXITER):
            clf.models['Hop{}_iter{}'.format(hopidx,iteidx)].save_model(saveroot+'CP_Hop{}_iter{}.txt'.format(hopidx,iteidx))
        
   
        
   
    #%%
    print('Ensemble Imagelevel')
    
    clf_ensemble = {}
    tr_prob_hop = []
    te_prob_hop = []
    
    tr_acc_all = []
    te_acc_all = []
    
    for i in range(len(HOPLIST)):
        tr_prob = clf.tr_prob_save[i][-1]
        print('Hop{} tr_prob_shape = {}'.format(i, tr_prob.shape))
        te_prob = clf.te_prob_save[i][-1]
        print('Hop{} te_prob_shape = {}'.format(i, te_prob.shape))
        # if i==0:
        #     # do not use border
        #     tt=1
        #     tr_prob = tr_prob[:,tt:(-1*tt),tt:(-1*tt)]
        #     te_prob = te_prob[:,tt:(-1*tt),tt:(-1*tt)]
        
        
        tr_prob = tr_prob.reshape(tr_prob.shape[0],-1)
        te_prob = te_prob.reshape(te_prob.shape[0],-1)
        tr_prob_hop.append(tr_prob)
        te_prob_hop.append(te_prob)
    
    tr_prob_hop = np.concatenate(tr_prob_hop,axis=1)#[:,1::2]
    te_prob_hop = np.concatenate(te_prob_hop,axis=1)#[:,1::2]
    print(tr_prob_hop.shape)
    print(te_prob_hop.shape)
    
    clf_ensemble['img'] = xgb.XGBClassifier(n_jobs=-1,
                                            objective='multi:softprob',
                                            tree_method='gpu_hist', gpu_id=FLAGS.GPU,
                                            max_depth=5,n_estimators=500,
                                            min_child_weight=10,gamma=5,
                                            subsample=1.0,learning_rate=0.1,
                                            nthread=8,colsample_bytree=0.6).fit(tr_prob_hop, y_train.reshape(-1),
                                                                                early_stopping_rounds=100,eval_metric='mlogloss',
                                                                                xgb_model=None, #,
                                                                                eval_set=[(te_prob_hop, y_test.reshape(-1))])      
                                                                
    tr_prob_final = clf_ensemble['img'].predict_proba(tr_prob_hop)
    te_prob_final = clf_ensemble['img'].predict_proba(te_prob_hop)
    
    tr_prob_final = tr_prob_final.reshape(len(AUG_list), -1, 10)
    tr_prob_final = np.mean(tr_prob_final, axis=0)

    te_prob_final = te_prob_final.reshape(len(AUG_list), -1, 10)
    te_prob_final = np.mean(te_prob_final, axis=0)

    print(tr_prob_final.shape)
    print(te_prob_final.shape)
    
    tr_acc = np.sum(np.argmax(tr_prob_final,axis=1)==y_train[:50000].reshape(-1))/50000.
    te_acc = np.sum(np.argmax(te_prob_final,axis=1)==y_test[:10000].reshape(-1))/10000.
    tr_acc_all.append(tr_acc)
    te_acc_all.append(te_acc)
    
    TIME1 = time.time()
    print("----------------------------- %s seconds ------------------------" % (TIME1-START_TIME))
    print('te_acc')


    te_acc_top1 = cal_top2_acc(te_prob_final, y_test[:10000],top=1)
    print(te_acc_top1)

    te_acc_top2 = cal_top2_acc(te_prob_final, y_test[:10000],top=2)
    print(te_acc_top2)


    print('saving models ... ')
    
    
    fwrite = open(saveroot+'modelES_Ffus.pkl','wb')
    pickle.dump(clf_ensemble, fwrite, protocol=2)
    fwrite.close()
    
    with open(saveroot + 'tr_prob_M2A_Ffus.npy', 'wb') as f:
        np.save(f, tr_prob_final)
    with open(saveroot + 'te_prob_M2A_Ffus.npy', 'wb') as f:
        np.save(f, te_prob_final)

    # plt.figure()
    # plt.plot(tr_acc_all,'.-')
    # plt.plot(te_acc_all,'.-')
    # plt.grid('on')
    # plt.xticks(np.arange(len(te_acc_all)),['Hop2','Hop2merge','Hop3','Hop3merge','Hop4','Hop4merge','img'])
    # plt.savefig(saveroot+'acc_plots_Ffus.png')
    # plt.close()
    # print(te_acc_all)
    
    
    # with open(saveroot + 'te_prob_M2A.npy', 'wb') as f:
    #     np.save(f, te_prob_M2A)
                
    # with open(saveroot + 'te_prob_M2A.npy', 'wb') as f:
    #     np.save(f, te_prob_M2A)
        
    # with open(saveroot + 'te_acc_pairwise.npy', 'wb') as f:
    #     np.save(f, te_acc_pairwise)
    # with open(saveroot + 'te_acc_updated_all.npy', 'wb') as f:
    #     np.save(f, np.array(te_acc_updated_all))
        
    # plt.plot(te_acc_updated_all,'.-');plt.grid('on');plt.savefig(saveroot+'updated_acc.png');plt.close()
    
                
