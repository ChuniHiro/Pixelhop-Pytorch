# v2022.03.10 # added kmeans-based feature selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from lib.lib_cross_entropy import cross_entropy, cal_weighted_CE, cal_weighted_H
from tqdm import tqdm

def classwise_distribution_curve(X_projected, y, K=32, bound=None, title=''):
    bins = np.arange(X_projected.min(), X_projected.max() + 0.01, (X_projected.max() - X_projected.min()) / K)
    # bins = model[index[i]][0]
    classwise = []
    for c in range(10):
        tmp, _, _ = plt.hist(X_projected[y == c], bins=bins)
        classwise.append(tmp)
        plt.close()
    ####################2
    plt.figure()
    for c in range(10):
        plt.plot(classwise[c], label=str(c))
    if bound is not None:
        diff = np.abs(bins-bound)
        idx = np.argmin(diff)
        if bound>bins[idx]:
            plt.axvline(x=(bound-bins[idx])/(bins[idx+1]-bins[idx])+idx,linestyle='--')
        else:
            plt.axvline(x=(bound-bins[idx-1])/(bins[idx]-bins[idx-1])+idx-1,linestyle='--')

    plt.xticks(np.arange(bins.size)[::5], np.round(bins[::5], decimals=2))
    plt.legend()
    plt.suptitle('Class-wise distribution '+title)

    plt.show()
    return classwise


class Disc_Feature_Test():
    def __init__(self, num_class, num_Candidate=32, bin_mode='uniform', loss='cross_entropy'):
        self.num_class = (int)(num_class)
        # self.num_bin = (int)(num_bin)
        self.num_Candidate = (int)(num_Candidate)
        self.bin_mode = bin_mode
        self.loss = loss
        self.loss_list = []

    def lloyd_max(self,x_points,num_cluster = 33):
        # interval = (x_points.max() - x_points.min())/num_cluster
        # init = [x_points.min()+interval/2]
        # for i in range(num_cluster-1):
        #     init.append(init[i] + interval)
        # init = np.array(init).reshape(-1,1)

        init = np.arange(x_points.min(),x_points.max(),(x_points.max()-x_points.min())/num_cluster).reshape(-1,1)

        kmean = MiniBatchKMeans(n_clusters=num_cluster,init=init,batch_size=5000).fit(x_points.reshape(-1,1))
        centroids = kmean.cluster_centers_
        centroids = centroids.squeeze()

        centroids = np.sort(centroids)

        # generate boundary
        boundary = []
        # boundary.append(-1*np.float('inf'))
        for i in range(num_cluster-1):
            boundary.append((centroids[i]+centroids[i+1])/2)
        # boundary.append(np.float('inf'))

        if False:
            plt.hist(x_points.squeeze(), bins=boundary)
            # for i in range(1,len(boundary)):
            #     plt.axvline(x=boundary[i], color='orange')#, linestyle='--')
            plt.show()

        return np.sort(np.array(boundary)), kmean


    def bin_process(self,x,y):
        if np.max(x) ==  np.min(x):
            return np.zeros(x.shape[0]).astype('int64'), 1 #x.astype('int64')

        # B bins (B-1) candicates of partioning point
        B_ = self.num_Candidate
        if self.bin_mode == 'uniform':
            candidates = np.arange(np.min(x),np.max(x),(np.max(x)-np.min(x))/(B_))
            candidates = candidates[1:]
        elif self.bin_mode == 'kmean':
            candidates, _ = self.lloyd_max(x, num_cluster=B_)
        candidates = np.unique(candidates)

        loss_i = np.zeros(candidates.shape[0])
        if self.loss == 'cross_entropy':
            for idx in range(candidates.shape[0]):
                loss_i[idx] = cal_weighted_CE(x, y, candidates[idx],num_cls=self.num_class)
        elif self.loss == 'entropy':
            for idx in range(candidates.shape[0]):
                loss_i[idx] = cal_weighted_H(x, y, candidates[idx],num_cls=self.num_class)
        # elif self.loss == 'avg_entropy':
        #     for idx in range(candidates.shape[0]):
        #         loss_i[idx] = cal_avg_H(x, y, candidates[idx],num_cls=self.num_class)

        self.loss_list.append(loss_i)
        best_bound = candidates[np.argmin(loss_i)]
        best_loss = np.min(loss_i)

        # if True:
        #     classwise_distribution_curve(x,y,bound=best_bound)

        bin_labels = np.zeros(x.shape[0]).astype('int64')
        bin_labels[x<best_bound] = 0
        bin_labels[x>=best_bound] = 1
        return bin_labels, best_loss


    def loss_estimation(self, x, y): # 2022.03.12
        x = x.astype('float64')
        y = y.astype('int64')
        y = y.squeeze()
        _, minimum_loss = self.bin_process(x.squeeze(), y)
        return minimum_loss

    def get_all_loss(self, X, Y): # 2022.03.12
        '''
        Parameters
        ----------
        X : TYPE
            shape (N, M).
        Y : TYPE
            shape (N).

        Returns
        -------
        feat_ce: CE for all the feature dimensions. The smaller the better

        '''
        feat_ce = np.zeros(X.shape[-1])
        for k in tqdm(range(X.shape[-1])):
            feat_ce[k] = self.loss_estimation(X[:,[k]], Y)
        return feat_ce
