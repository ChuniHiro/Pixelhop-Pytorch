'''
v2021.07.07
Step 1b -- Preliminary Attention Statistics Learning
@Author: Yijing Yang
@Contact: yangyijing710@outlook.com
'''

import pickle
from keras.datasets import cifar10
import numpy as np
from tensorflow.python.platform import flags
import time
import os
from skimage.measure import block_reduce
from skimage.util import view_as_windows
import numba
import warnings
from lib.lib_pixelwiseCLF_2cls import pixelwiseCLF
warnings.filterwarnings("ignore")


flags.DEFINE_string("featroot", '../attPixelHopFeat/', "feature root") # Figure 1
flags.DEFINE_string("saveroot", '../saveAtt/', "saveroot")
flags.DEFINE_string("colorspace", 'RGB', "color space")
flags.DEFINE_string("HOPLIST", "3,4,5,6,7,8", "Hop unit to be included in the feature, 3 means Hop-4, etc.")
flags.DEFINE_string("AUG", "0", "augmentation mode, 0 means no augmentation")
flags.DEFINE_integer("NODC", 0, "discard DC or not")
flags.DEFINE_integer("MAXITER", 1, "max iteration number for label smoothing. 1 means no label smoothing")
flags.DEFINE_integer("pair0",3, "binary confusing set class #1")
flags.DEFINE_integer("pair1",5, "binary confusing set class #1")
flags.DEFINE_string("GPU","0", "GPU ID")
flags.DEFINE_integer("POOL", 2, "pooling")
flags.DEFINE_float("SELECT", 1.0, "dimension/percentage of feature selection")
flags.DEFINE_string("FStype", 'ANOVA', "Feature selection mode: CE, ANOVA, XGB")
FLAGS = flags.FLAGS

FStype = FLAGS.FStype
if FLAGS.colorspace == 'PQR':
    root = [FLAGS.featroot + 'P/', FLAGS.featroot + 'Q/', FLAGS.featroot + 'PQR_R/']
else:
    root = [FLAGS.featroot + '/']

saveroot = FLAGS.saveroot + '/pair{}_{}/'.format(FLAGS.pair0, FLAGS.pair1)
if not os.path.isdir(saveroot): os.makedirs(saveroot)

AUG_list = [int(s) for s in FLAGS.AUG.split(',')]
HOPLIST = [int(s) for s in FLAGS.HOPLIST.split(',')]


@numba.jit(forceobj=True, parallel=True)
def remove_std(X: np.ndarray, STD: np.ndarray):
    return X / STD


@numba.jit(forceobj=True, parallel=True)
def cal_std(X: np.ndarray):
    return np.std(X, axis=0, keepdims=1).reshape(1, 1, 1, -1)


def get_feat_center_region(hopidx=0, mode='tr', ROOT=None, chosenidx=None, aug_idx=0):
    if hopidx < HOPLIST[-1]:
        marg = HOPLIST[-1]

        if chosenidx is None:
            if mode == 'tr':
                with open(ROOT + 'tr_feature_Hop' + str(hopidx + 1) + '_AUG' + str(aug_idx) + '.npy', 'rb') as f:
                    X_train_all = np.load(f)[:, (marg - hopidx):(hopidx - marg), (marg - hopidx):(hopidx - marg),
                                  FLAGS.NODC:]
                return X_train_all
            else:
                with open(ROOT + 'te_feature_Hop' + str(hopidx + 1) + '_AUG' + str(aug_idx) + '.npy', 'rb') as f:
                    X_test_all = np.load(f)[:, (marg - hopidx):(hopidx - marg), (marg - hopidx):(hopidx - marg),
                                 FLAGS.NODC:]
                return X_test_all
        else:
            if mode == 'tr':
                with open(ROOT + 'tr_feature_Hop' + str(hopidx + 1) + '_AUG' + str(aug_idx) + '.npy', 'rb') as f:
                    X_train_all = \
                    np.load(f)[:, (marg - hopidx):(hopidx - marg), (marg - hopidx):(hopidx - marg), FLAGS.NODC:][
                        chosenidx]
                return X_train_all
            else:
                with open(ROOT + 'te_feature_Hop' + str(hopidx + 1) + '_AUG' + str(aug_idx) + '.npy', 'rb') as f:
                    X_test_all = \
                    np.load(f)[:, (marg - hopidx):(hopidx - marg), (marg - hopidx):(hopidx - marg), FLAGS.NODC:][
                        chosenidx]
                return X_test_all
    else:
        if chosenidx is None:
            if mode == 'tr':
                with open(ROOT + 'tr_feature_Hop' + str(hopidx + 1) + '_AUG' + str(aug_idx) + '.npy', 'rb') as f:
                    X_train_all = np.load(f)[:, :, :, FLAGS.NODC:]
                return X_train_all
            else:
                with open(ROOT + 'te_feature_Hop' + str(hopidx + 1) + '_AUG' + str(aug_idx) + '.npy', 'rb') as f:
                    X_test_all = np.load(f)[:, :, :, FLAGS.NODC:]
                return X_test_all
        else:
            if mode == 'tr':
                with open(ROOT + 'tr_feature_Hop' + str(hopidx + 1) + '_AUG' + str(aug_idx) + '.npy', 'rb') as f:
                    X_train_all = np.load(f)[:, :, :, FLAGS.NODC:][chosenidx]
                return X_train_all
            else:
                with open(ROOT + 'te_feature_Hop' + str(hopidx + 1) + '_AUG' + str(aug_idx) + '.npy', 'rb') as f:
                    X_test_all = np.load(f)[:, :, :, FLAGS.NODC:][chosenidx]
                return X_test_all


def load_feat_allAUG_singleHop(AUG_list, mode='tr', hop=0, chosen=None):
    tr_X = []
    for aug in AUG_list:
        print('aug{}'.format(aug))
        tr_X.append(get_feat_center_region(hopidx=hop, mode=mode, ROOT=root[0], aug_idx=aug, chosenidx=chosen) )
    tr_X = np.concatenate(tr_X, axis=0)

    ch = tr_X.shape[-1]
    if True:
        if FLAGS.POOL == 2:
            tr_X = block_reduce(tr_X, (1, FLAGS.POOL, FLAGS.POOL, 1), np.max)
        elif FLAGS.POOL == 3:  # overlapping pool
            tr_X = view_as_windows(tr_X, (1, FLAGS.POOL, FLAGS.POOL, ch), (1, 2, 2, ch))
            tr_X = tr_X.reshape(tr_X.shape[0], tr_X.shape[1], tr_X.shape[2], FLAGS.POOL * FLAGS.POOL, -1)
            tr_X = np.max(tr_X, axis=3)

    return tr_X


def logical_AND_multi(log1, log2, *args):
    length = len(args)
    indicator = log1 * log2
    for i in range(length):
        indicator *= args[i]
    return indicator


def logical_OR_multi(log1, log2, *args):
    length = len(args)
    indicator = log1 + log2
    for i in range(length):
        indicator += args[i]
    return indicator


def get_groups_TRAIN_2(tr_y, PAIR0, PAIR1):
    tr_chosen = logical_OR_multi(tr_y == PAIR0, tr_y == PAIR1).squeeze()
    ynew = np.copy(tr_y[tr_chosen])
    ynew[ynew == PAIR0] = 0
    ynew[ynew == PAIR1] = 1
    print(np.unique(ynew))

    return ynew, tr_chosen


if __name__ == '__main__':
    NUM_CLS = 2

    # read image data
    (train_images, y_train), (test_images, y_test) = cifar10.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    totalN = train_images.shape[0]

    y_train = y_train.squeeze().astype('int')
    y_test = y_test.squeeze().astype('int')
    if len(AUG_list) > 1:
        y_train = np.tile(y_train, len(AUG_list))
        y_test = np.tile(y_test, len(AUG_list))

    # select a binary confusing set (training samples) of class [pair0] and class [pair1]
    y_train_PAIR, tr_chosen = get_groups_TRAIN_2(y_train, FLAGS.pair0, FLAGS.pair1)

    # load feature
    print('Context vector extraction >>')
    tr_X_ss_list, te_X_ss_list = [], []
    selected_idx_all = []
    STD_list = []
    for i in range(len(HOPLIST)):
        # load feature as tr_X_ss
        print('Hop{}... loading feature'.format(i))
        tr_X_ss = load_feat_allAUG_singleHop(AUG_list, mode='tr', hop=HOPLIST[i], chosen=tr_chosen[:totalN])
        tr_NN, HH, WW, CC = tr_X_ss.shape

        # generate pixel-wise label
        y_train_pix_list = np.repeat(y_train_PAIR, HH * WW)

        # remove STD
        STD = cal_std(tr_X_ss.reshape(-1, CC))
        tr_X_ss = remove_std(tr_X_ss, STD)
        STD_list.append(STD)

        # % Feature selection
        # if FLAGS.SELECT < 1.0:
        #     print('feature select')
        #     tr_X_ss, selected_idx, _ = FEAT.feature_selection_train(tr_X_ss.reshape(-1, CC), y_train_pix_list,
        #                                                             FStype='ANOVA', thrs=FLAGS.SELECT)
        #     selected_idx_all.append(selected_idx)

        tr_X_ss_list.append(tr_X_ss.reshape(tr_NN, HH, WW, -1))

    # concatenate features in the center 14x14 region
    tr_X_ss_list = np.concatenate(tr_X_ss_list, axis=-1)
    print('>>>>>>>>>>>>>>>>>>')

    # ============================== Get confidence score =============================
    START_TIME = time.time()
    print('Get confidence level >>')

    tr_prob_list, te_prob_list = [], []
    sampWeight = None
    chosenidx = None

    for ite in range(FLAGS.MAXITER): # if MAXITER>1, using soft-label smoothing. Default should be 1
        clf = pixelwiseCLF(numcls=2, GPU=FLAGS.GPU, regC=1.0, standardize=False, model_select=False)
        clf.fit(tr_X_ss_list, y_train_PAIR, sample_weight=sampWeight, chosen=chosenidx)
        tr_prob_hop_ite, _ = clf.predict_proba(tr_X_ss_list, y=y_train_PAIR, reduce=0)
        tr_prob_list.append(tr_prob_hop_ite)


    fwrite = open(saveroot + 'tr_prob_list.pkl', 'wb')
    pickle.dump(tr_prob_list, fwrite, protocol=2)
    fwrite.close()


    print('finish')
