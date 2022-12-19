'''
v2021.07.07
Step 2 -- Attention Map Refinement
@Author: Yijing Yang
@Contact: yangyijing710@outlook.com
'''

import pickle
from keras.datasets import cifar10
import numpy as np
from lib.seg_utils import up_stream
import xgboost as xgb
from tensorflow.python.platform import flags
import os
import lib.attAnnotations as ANNO
from skimage.measure import block_reduce
from skimage.util import view_as_windows
import numba
import warnings
warnings.filterwarnings("ignore")


flags.DEFINE_string("featroot", '../attRefinementFeat/', "feature root") # Figure 3
flags.DEFINE_string("saveroot", '../saveAtt/', "saveroot")
flags.DEFINE_string("subroot", '0824_v1', "sub-root")
flags.DEFINE_string("colorspace", 'RGB', "color space")
flags.DEFINE_string("HOPLIST","1,3", "Hop unit to be included in the feature, 1 means Hop-2, 3 means Hop-4")
flags.DEFINE_string("AUG","0", "augmentation mode, 0 means no augmentation")
flags.DEFINE_integer("NODC",0, "discard DC or not")
flags.DEFINE_integer("pair0",3, "binary confusing set class #1")
flags.DEFINE_integer("pair1",5, "binary confusing set class #1")
flags.DEFINE_string("GPU","0", "GPU ID")
flags.DEFINE_integer("POOL",0, "Pooling")
flags.DEFINE_integer("tr_att_mode",1, "The training images selection mode. 1: selected, 2: random")

FLAGS = flags.FLAGS

AUG_list = [int(s) for s in FLAGS.AUG.split(',')]
HOPLIST = [int(s) for s in FLAGS.HOPLIST.split(',')]

if FLAGS.colorspace =='PQR':
    root = [FLAGS.root+'P/',FLAGS.root+'Q/',FLAGS.root+'PQR_R/']
else:
    root = [FLAGS.root + '/']

saveroottmp = FLAGS.saveroot + '/pair{}_{}/'.format(FLAGS.pair0, FLAGS.pair1)
saverootBBox = saveroottmp + 'trainBBox/'
saveroot = saveroottmp + FLAGS.subroot+'/'

if not os.path.isdir(saveroot):os.makedirs(saveroot)

@numba.jit(forceobj = True, parallel = True)
def remove_std(X: np.ndarray, STD: np.ndarray):
    return X/STD

@numba.jit(forceobj = True, parallel = True)
def cal_std(X: np.ndarray):
    return np.std(X, axis=0, keepdims=1).reshape(1,1,1,-1)

def get_feat_grouping_raw(hopidx=0, target=[3, 5], mode='tr', ROOT=None, chosenidx=None, aug_idx=0):
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
        tr_X.append(get_feat_grouping_raw(hopidx=hop, mode=mode, ROOT=root[0], aug_idx=aug, chosenidx=chosen) )
    tr_X = np.concatenate(tr_X, axis=0)

    ch = tr_X.shape[-1]
    if FLAGS.POOL == 2:
        tr_X = block_reduce(tr_X, (1, FLAGS.POOL, FLAGS.POOL, 1), np.max)
    elif FLAGS.POOL == 3:  # overlapping pool
        tr_X = view_as_windows(tr_X, (1, FLAGS.POOL, FLAGS.POOL, ch), (1, 2, 2, ch))
        tr_X = tr_X.reshape(tr_X.shape[0], tr_X.shape[1], tr_X.shape[2], FLAGS.POOL * FLAGS.POOL, -1)
        tr_X = np.max(tr_X, axis=3)

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

def get_groups_TRAIN_2(tr_y, PAIR0, PAIR1):
    tr_chosen = logical_OR_multi(tr_y == PAIR0, tr_y == PAIR1).squeeze()
    ynew = np.copy(tr_y[tr_chosen])
    ynew[ynew==PAIR0] = 0
    ynew[ynew==PAIR1] = 1
    print(np.unique(ynew))
    return ynew, tr_chosen


if __name__=='__main__':
    NUM_CLS = 2

    # read image data
    (train_images, y_train), (test_images, y_test) = cifar10.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    totalN = train_images.shape[0]

    class_name_all = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    class_name = [class_name_all[FLAGS.pair0], class_name_all[FLAGS.pair1]]

    y_train = y_train.squeeze().astype('int')
    y_test = y_test.squeeze().astype('int')

    if len(AUG_list) > 1:
        y_train = np.tile(y_train, len(AUG_list))
        y_test = np.tile(y_test, len(AUG_list))

    y_train_PAIR, tr_chosen = get_groups_TRAIN_2(y_train, FLAGS.pair0, FLAGS.pair1)
    y_test_PAIR, te_chosen = get_groups_TRAIN_2(y_test, FLAGS.pair0, FLAGS.pair1)

    train_images = train_images[tr_chosen]
    test_images = test_images[te_chosen]

    # load preliminary attention window information
    fread = open(saverootBBox + 'tr_bbox.pkl', 'rb')
    tr_bbox = pickle.load(fread)
    fread.close()

    # ============================== Select a small subset for training =============================
    # select training samples for attention map refinement
    if FLAGS.tr_att_mode == 1: # selected good preliminary results
        tr_selectidx_0 = ANNO.example_att_list(FLAGS.pair0)
        tr_selectidx_1 = ANNO.example_att_list(FLAGS.pair1)
    elif FLAGS.tr_att_mode == 2: # select randomly
        tr_selectidx_0 = ANNO.random_att_list(y_train_PAIR, FLAGS.pair0, num=25)
        tr_selectidx_1 = ANNO.random_att_list(y_train_PAIR, FLAGS.pair1, num=25)
    # elif FLAGS.tr_att_mode == 3: # select high DFT
    #     tr_selectidx_0 = ANNO.dft_att_list(FLAGS.pair0)
    #     tr_selectidx_1 = ANNO.dft_att_list(FLAGS.pair1)

    tr_selectidx = tr_selectidx_0 + tr_selectidx_1
    tr_selectidx = np.unique(np.array(tr_selectidx))
    print(tr_selectidx.size)

    # ============================== Feature aggragation for refinement learning =============================
    # Load Hop-2 and Hop-4 feature (hop=1, 3)
    tr_X_ss_list = []
    for i in range(len(HOPLIST)):
        tmp = load_feat_allAUG_singleHop(AUG_list, mode='tr', hop=HOPLIST[i], chosen=tr_chosen)
        abs_max = np.max(np.abs(tmp))
        tmp = tmp / abs_max
        tr_X_ss_list.append(tmp)

    # Upsampling & Concatenation
    tr_X_ss_list = up_stream(tr_X_ss_list, num_layer=len(tr_X_ss_list), interp='bilinear')
    print(tr_X_ss_list.shape)

    # Get foreground mask and the cooresponding feature
    fore_mask = np.zeros((len(tr_selectidx), 32,32))
    for i in range(len(tr_selectidx)):
        # -1,1,0 means discarded region, positive samples (foreground), and negative samples (background)
        fore_mask[i][(tr_bbox['UL'][i, 0]-2):(tr_bbox['LL'][i, 0]+2), (tr_bbox['UL'][i, 1]-2):(tr_bbox['UR'][i, 1]+2)] = -1
        fore_mask[i][(tr_bbox['UL'][i, 0]+1):(tr_bbox['LL'][i, 0]-1), (tr_bbox['UL'][i, 1]+1):(tr_bbox['UR'][i, 1]-1)] = 1
    fwrite = open(saveroot + 'fore_mask.pkl', 'wb')
    pickle.dump(fore_mask, fwrite)
    fwrite.close()

    fore_feat = tr_X_ss_list[tr_selectidx][fore_mask == 1].reshape(-1, tr_X_ss_list.shape[-1])

    # Get background mask and the cooresponding feature
    back_idx = np.random.permutation(np.sum(fore_mask == 0))[:fore_feat.shape[0]] # same sample number as foreground
    back_feat = tr_X_ss_list[tr_selectidx][fore_mask == 0].reshape(-1, tr_X_ss_list.shape[-1])[back_idx]

    # Gather the foreground (1) and background (0) samples and labels
    X_FB = np.concatenate((fore_feat, back_feat), axis=0)
    y_FB = np.zeros(X_FB.shape[0])
    y_FB[:fore_feat.shape[0]] = 1
    del fore_feat, back_feat

    # shuffle
    randidx = np.random.permutation(y_FB.size)
    X_FB = X_FB[randidx]
    y_FB = y_FB[randidx]

    # ============================== Training  =============================
    # classifier for foreground and background
    clf = xgb.XGBClassifier(n_jobs=6, objective="binary:logistic",
                            max_depth=6, n_estimators=300,
                            tree_method='gpu_hist', gpu_id=FLAGS.GPU,
                            min_child_weight=5, gamma=5,
                            subsample=0.8, learning_rate=0.1,
                            nthread=6, colsample_bytree=1.0).fit(X_FB, y_FB)
    # predict the binary mask
    tr_out = clf.predict_proba(tr_X_ss_list.reshape(-1, tr_X_ss_list.shape[-1]))[:, 1]
    tr_out = tr_out.reshape(tr_X_ss_list.shape[0], tr_X_ss_list.shape[1], tr_X_ss_list.shape[2])

    fwrite = open(saveroot + 'tr_refined_att.pkl', 'wb')
    pickle.dump(tr_out, fwrite)
    fwrite.close()

    del tr_X_ss_list

    # ============================== Testing  =============================
    # load test feature
    te_X_ss_list = []
    for i in range(len(HOPLIST)):
        tmp = load_feat_allAUG_singleHop(AUG_list, mode='te', hop=HOPLIST[i], chosen=te_chosen)
        abs_max = np.max(np.abs(tmp))
        tmp = tmp / abs_max
        te_X_ss_list.append(tmp)
    te_X_ss_list = up_stream(te_X_ss_list, num_layer=len(te_X_ss_list), interp='bilinear')

    # predict the binary mask
    te_out = clf.predict_proba(te_X_ss_list.reshape(-1, te_X_ss_list.shape[-1]))[:, 1]
    te_out = te_out.reshape(te_X_ss_list.shape[0], te_X_ss_list.shape[1], te_X_ss_list.shape[2])

    fwrite = open(saveroot + 'te_refined_att.pkl', 'wb')
    pickle.dump(te_out, fwrite)
    fwrite.close()

    print('finish')