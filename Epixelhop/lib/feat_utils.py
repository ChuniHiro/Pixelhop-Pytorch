# v2022.03.17 Yijing
import numpy as np
import lib.lib_stats as STAT
from sklearn.feature_selection import f_classif

def feature_selection(tr_X, tr_y, FStype='DFT_entropy',thrs=1.0,B=32): #03.11.2022
    NUM_CLS = np.unique(tr_y).size
    dft_loss_list = []
    if FStype == 'DFT_entropy': # lower the better (currently used one 03.17.2022)
        dft = STAT.Disc_Feature_Test(num_class=NUM_CLS, num_Candidate=B, bin_mode='uniform',loss='entropy')
        feat_score = dft.get_all_loss(tr_X, tr_y)
        feat_sorted_idx = np.argsort(feat_score)
        dft_loss_list = np.array(dft.loss_list) # the list of loss for all the (B-1 candidates)

    elif FStype == 'DFT_uniform': # lower the better
        dft = STAT.Disc_Feature_Test(num_class=NUM_CLS, num_Candidate=B, bin_mode='uniform',loss='cross_entropy')
        feat_score = dft.get_all_loss(tr_X, tr_y)
        feat_sorted_idx = np.argsort(feat_score)
        dft_loss_list = np.array(dft.loss_list)

    elif FStype == 'DFT_km':  # lower the better
        dft = STAT.Disc_Feature_Test(num_class=NUM_CLS, num_Candidate=B, bin_mode='kmean', loss='cross_entropy')
        feat_score = dft.get_all_loss(tr_X, tr_y)
        feat_sorted_idx = np.argsort(feat_score)
        dft_loss_list = np.array(dft.loss_list)

    # elif FStype == 'DFT_AvgEntropy': # lower the better
    #     dft = STAT.Disc_Feature_Test(num_class=NUM_CLS, num_bin=numbin, num_Candidate=B, bin_mode='uniform',loss='avg_entropy')
    #     feat_score = dft.get_all_loss(tr_X, tr_y)
    #     feat_sorted_idx = np.argsort(feat_score)
    #     dft_loss_list = np.array(dft.loss_list)
    # elif FStype == 'DFT_acc': # higher the better
    #     dft = STAT.Disc_Feature_Test(num_class=NUM_CLS, num_bin=numbin, num_Candidate=B, bin_mode='uniform',loss='acc')
    #     feat_score = dft.get_all_loss(tr_X, tr_y)
    #     feat_sorted_idx = np.argsort(feat_score)
    #     dft_loss_list = np.array(dft.loss_list)

    elif FStype == 'anova': # higher the better
        feat_score, _ = f_classif(tr_X, tr_y)
        feat_sorted_idx = np.argsort(-1*feat_score)

    elif FStype == 'corr': # higher the better
        feat_score = np.abs(np.corrcoef(tr_X, tr_y.reshape(-1, 1), rowvar=False)[-1, :-1])
        feat_sorted_idx = np.argsort(-1*feat_score)

    elif FStype == 'var': # higher the better
        feat_score = np.var(tr_X, axis=0)
        feat_sorted_idx = np.argsort(-1*feat_score)

    selected_idx = feat_sorted_idx[:int(thrs*feat_score.size)]

    return selected_idx, feat_score, dft_loss_list


def trans_feat(tr_X, te_X, fmode='norm'):
    if fmode == 'norm' or 'energy':
        MAX_ = np.max(tr_X, axis=0, keepdims=1)

        tr_X_new = tr_X / MAX_
        te_X_new = te_X / MAX_

        if fmode == 'energy':
            tr_X_new = tr_X_new ** 2
            te_X_new = te_X_new ** 2

    return tr_X_new, te_X_new

if __name__ == '__main__':
    from keras.datasets import cifar10, mnist, fashion_mnist
    (train_images, y_train), (test_images, y_test) = mnist.load_data()

    tr_feat = train_images.reshape(60000,-1)
    _, dft_feat_loss, dft_loss_list = feature_selection(tr_feat, y_train, FStype='DFT_entropy', thrs=1.0, B=32)

    print('finished')
