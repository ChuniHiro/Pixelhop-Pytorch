'''
v2021.07.07
Step 3 -- Attention region finalization
@Author: Yijing Yang
@Contact: yangyijing710@outlook.com
'''

import pickle
from keras.datasets import cifar10
import numpy as np
from tensorflow.python.platform import flags
import os
import lib.lib_attention as ATT
import warnings
warnings.filterwarnings("ignore")

flags.DEFINE_string("saveroot", '../saveAtt/', "saveroot")
flags.DEFINE_string("subroot", '0824_v1', "sub-root")
flags.DEFINE_string("AUG","0", "augmentation mode")
flags.DEFINE_float("thrs",0.5, "Binarization threshold")
flags.DEFINE_integer("pair0",3, "augmentation mode")
flags.DEFINE_integer("pair1",5, "augmentation mode")

FLAGS = flags.FLAGS

AUG_list = [int(s) for s in FLAGS.AUG.split(',')]

saveroottmp = FLAGS.saveroot + '/pair{}_{}/'.format(FLAGS.pair0, FLAGS.pair1)
saverootBBox = saveroottmp + 'trainBBox/'
saverootRefine = saveroottmp + FLAGS.subroot+'/'
saveroot_cropped = saverootRefine + 'BBOX_adaptive_v2_thrs{}/'.format(FLAGS.thrs)

if not os.path.isdir(saveroot_cropped):os.makedirs(saveroot_cropped)

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

def normalize(data, ABS=False):
    if ABS:
        data_new = np.abs(data)
    else:
        data_new = data
    data_new = data_new - data_new.min()
    return data_new/(data_new.max()+1e-5)

def att_bbox_finalization(images, heatmap):
    cropped = []
    for n in range(heatmap.shape[0]):
        print(n)
        heat = normalize(heatmap[n].astype('float32'))
        bbox, _ = ATT.heat2bbox_adaptive_v2(heat, thrs=FLAGS.thrs, num_bb=1)
        cropped_n = normalize(ATT.crop_att(images[n], bbox))
        cropped.append(cropped_n)
    cropped = np.array(cropped)
    return cropped

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

    # ============================== Attention region finalization using refined heatmap =============================
    # training images
    fread = open(saverootRefine + 'tr_refined_att.pkl', 'rb')
    tr_refined_att = pickle.load(fread)
    fread.close()

    tr_cropped = att_bbox_finalization(train_images, tr_refined_att)

    fwrite = open(saveroot_cropped + 'tr_cropped.pkl', 'wb')
    pickle.dump(tr_cropped, fwrite)
    fwrite.close()


    # test images
    fread = open(saverootRefine + 'te_refined_att.pkl', 'rb')
    te_refined_att = pickle.load(fread)
    fread.close()

    te_cropped = att_bbox_finalization(test_images, te_refined_att)

    fwrite = open(saveroot_cropped + 'te_cropped.pkl', 'wb')
    pickle.dump(te_cropped, fwrite)
    fwrite.close()


    print('finish')