'''
v2021.07.07
Step 1b -- Save Bounding Box Coordinates
@Author: Yijing Yang
@Contact: yangyijing710@outlook.com
'''

import pickle
from keras.datasets import cifar10
import numpy as np
from tensorflow.python.platform import flags
import os
import numba
import warnings
warnings.filterwarnings("ignore")

flags.DEFINE_string("featroot", '../attPixelHopFeat/', "feature root") # Figure 1
flags.DEFINE_string("saveroot", '../saveAtt/', "saveroot")
flags.DEFINE_string("colorspace", 'RGB', "color space")
flags.DEFINE_string("AUG", "0", "augmentation mode, 0 means no augmentation")
flags.DEFINE_integer("pair0",3, "binary confusing set class #1")
flags.DEFINE_integer("pair1",5, "binary confusing set class #1")
FLAGS = flags.FLAGS

if FLAGS.colorspace == 'PQR':
    root = [FLAGS.featroot + 'P/', FLAGS.featroot + 'Q/', FLAGS.featroot + 'PQR_R/']
else:
    root = [FLAGS.featroot + '/']

saveroot = FLAGS.saveroot + '/pair{}_{}/'.format(FLAGS.pair0, FLAGS.pair1)
saverootBBox = saveroot+'trainBBox/'
if not os.path.isdir(saveroot):os.makedirs(saveroot)
if not os.path.isdir(saverootBBox):os.makedirs(saverootBBox)

AUG_list = [int(s) for s in FLAGS.AUG.split(',')]


@numba.jit(forceobj = True, parallel = True)
def remove_std(X: np.ndarray, STD: np.ndarray):
    return X/STD

@numba.jit(forceobj = True, parallel = True)
def cal_std(X: np.ndarray):
    return np.std(X, axis=0, keepdims=1).reshape(1,1,1,-1)


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


def generate_AttWindow(prob_n, win=19): # topk: number of bbox
    '''
    Attention Window Selection
    prob_n: The pixel-wise predicted probability vectors (H,W,C)
    win: Receptive field size of the attention window
    '''

    # TARGET = prob_n.shape[1]*2
    for iteridx in range(1):
        dft_LS = np.abs(np.max(prob_n,axis=-1) - 1.0/NUM_CLS)
        # dft_HS = cv2.resize(dft_LS, (TARGET,TARGET), interpolation=cv2.INTER_LINEAR)

    # sort the DFT scores of all the pixels
    sorted_idx = np.argsort(-1 * dft_LS.reshape(-1))
    dft_LS_flatten = dft_LS.reshape(-1)

    bbox_all = []
    topK = 1
    for k in range(topK):
        x_idx, y_idx = np.unravel_index(sorted_idx[k], dft_LS.shape)

        # approximate the doubled resolution
        x_idx = x_idx*2
        y_idx = y_idx*2

        x_idx += win//2
        y_idx += win//2

        # save the bounding box locations
        bbox = {}
        bbox['UL'] = [x_idx - win//2, y_idx - win//2] # uppoer left corner
        bbox['UR'] = [x_idx - win//2, y_idx + win//2] # upper right corner
        bbox['LL'] = [x_idx + win//2, y_idx - win//2] # lower left corner
        bbox['LR'] = [x_idx + win//2, y_idx + win//2] # lower right corner
        bbox['dft'] = np.copy(dft_LS_flatten[sorted_idx[k]])

        bbox_all.append(bbox)

    return bbox_all[0]


if __name__=='__main__':
    NUM_CLS = 2

    # read image data
    (train_images, y_train), (test_images, y_test) = cifar10.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    totalN = train_images.shape[0]

    class_name_all=['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
    class_name = [class_name_all[FLAGS.pair0], class_name_all[FLAGS.pair1]]

    y_train = y_train.squeeze().astype('int')
    y_test = y_test.squeeze().astype('int')

    if len(AUG_list)>1:
        y_train = np.tile(y_train, len(AUG_list))
        y_test = np.tile(y_test, len(AUG_list))

    # select a binary confusing set (training samples) of class [pair0] and class [pair1]
    y_train_PAIR, tr_chosen = get_groups_TRAIN_2(y_train, FLAGS.pair0, FLAGS.pair1)
    train_images = train_images[tr_chosen]

    # load the confidence scores for each pixel
    fread = open(saveroot+'tr_prob_list.pkl','rb')
    tr_prob_list = pickle.load(fread)
    fread.close()
    # By default, use the first iteration (without label smoothing)
    iteridx = 0
    tr_prob_hop_i = tr_prob_list[iteridx]

    # ============================== Extract Preliminary Attention Window =============================
    tr_bbox = {}
    tr_bbox['UL'] = []
    tr_bbox['UR'] = []
    tr_bbox['LL'] = []
    tr_bbox['LR'] = []
    tr_bbox['dft'] = []

    recepSize = 19 # the receptive field of the current setting is 19x19
    for n in range(tr_prob_hop_i.shape[0]):
        bbox = generate_AttWindow(tr_prob_hop_i[n], win=recepSize)
        tr_bbox['UL'].append(bbox['UL'])
        tr_bbox['UR'].append(bbox['UR'])
        tr_bbox['LL'].append(bbox['LL'])
        tr_bbox['LR'].append(bbox['LR'])
        tr_bbox['dft'].append(bbox['dft'])

    tr_bbox['UL'] = np.array(tr_bbox['UL'])
    tr_bbox['UR'] = np.array(tr_bbox['UR'])
    tr_bbox['LL'] = np.array(tr_bbox['LL'])
    tr_bbox['LR'] = np.array(tr_bbox['LR'])
    tr_bbox['dft'] = np.array(tr_bbox['dft'])

    fwrite = open(saverootBBox + 'tr_bbox.pkl', 'wb')
    pickle.dump(tr_bbox, fwrite, protocol=2)
    fwrite.close()

    print('finish')