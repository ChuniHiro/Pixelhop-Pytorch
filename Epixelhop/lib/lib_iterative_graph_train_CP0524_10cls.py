# v 2021.05.24
# train using incremental
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import log_loss
from skimage.util.shape import view_as_windows
import cv2

######
'''
work for 10 class for now
Construct graph aggregation
'''

class iter_clf():
    def __init__(self, agg = 1, max_iter=5, rank_neighbour=True, incre_train=False, models=None, small_model=False, eval_metric='mlogloss', version=7, gpu=None, saabfeat=True, neigh4=True,average =True, parent=True):
        self.eval = eval_metric # error logloss auc
        self.max_iter = max_iter
        self.gpu = gpu
        self.agg = agg
        self.rank_neighbour = rank_neighbour
        self.incre_train = incre_train
        self.saabfeat = saabfeat
        # self.Hop = Hop
        self.average = average
        self.parent = parent
        self.version = version
        self.neigh4 = neigh4
        self.small_model = small_model

        # self.prob = []
        self.tr_prob = []
        self.te_prob = []
        self.tr_prob_save = []
        self.te_prob_save = []
        
        
        self.tr_loss = []
        self.te_loss = []
        self.tr_acc = []
        self.te_acc = []
        self.xgbloss = []
        self.feature_imp = []
        
        self.XGB_previous_ = [None]
        if models is None:
            self.models = {}
            self.model_ready = False
        else:
            self.models= models
            self.model_ready = True
        self.NUM_ROUNDS = [0]
        
        
    def fit(self,tr_X, tr_Y, te_X, te_Y):
        self.numHop = len(tr_X)
        # tr_X = shape(N,H,W,C)
        self.H, self.W, self.C = [],[],[]
        
        for hop in range(self.numHop):
            self.tr_N, H, W, C = tr_X[hop].shape
            self.H.append(H); self.W.append(W); self.C.append(C)
            self.tr_acc.append([]); self.te_acc.append([]);
            self.tr_loss.append([]); self.te_loss.append([]);
            self.tr_prob.append([]); self.te_prob.append([]);
            self.tr_prob_save.append([]); self.te_prob_save.append([]);

        self.te_N = te_X[0].shape[0]

        for ite in range(self.max_iter):
            for hop in range(self.numHop):
                print('Iteration {}, Hop {}'.format(ite, hop))

                if ite==0 and (hop==0 or hop<self.numHop-1):
                    self.tr_prob[hop] = 0.1*np.ones((self.tr_N, self.H[hop], self.W[hop], 9))
                    self.te_prob[hop] = 0.1*np.ones((self.te_N, self.H[hop], self.W[hop], 9))
                    if self.parent and (hop<self.numHop-1):
                        self.tr_prob[hop+1] = 0.1*np.ones((self.tr_N, self.H[hop+1], self.W[hop+1], 9))
                        self.te_prob[hop+1] = 0.1*np.ones((self.te_N, self.H[hop+1], self.W[hop+1], 9))
                    
                if hop==0: # first stage don't use children (no children)
                    if (self.parent and ite>0):
                        tr_probmap_parent = self.gather_parent(self.tr_prob[hop+1], hopidx=hop).reshape(self.tr_N, self.H[hop], self.W[hop], -1)
                        tr_probmap_sibling = self.gather_neighbour(self.tr_prob[hop],hopidx=hop).reshape(self.tr_N, self.H[hop], self.W[hop], -1)
                        te_probmap_parent = self.gather_parent(self.te_prob[hop+1], hopidx=hop).reshape(self.te_N, self.H[hop], self.W[hop], -1)
                        te_probmap_sibling = self.gather_neighbour(self.te_prob[hop],hopidx=hop).reshape(self.te_N, self.H[hop], self.W[hop], -1)
                        
                        tr_probmap_agg = np.concatenate((tr_probmap_parent,tr_probmap_sibling),axis=-1)
                        te_probmap_agg = np.concatenate((te_probmap_parent,te_probmap_sibling),axis=-1)
                    else:
                        tr_probmap_agg = self.gather_neighbour(self.tr_prob[hop],hopidx=hop).reshape(self.tr_N, self.H[hop], self.W[hop], -1)
                        te_probmap_agg = self.gather_neighbour(self.te_prob[hop],hopidx=hop).reshape(self.te_N, self.H[hop], self.W[hop], -1)
                else:
                    tr_probmap_child = self.gather_children(self.tr_prob[hop-1],hopidx=hop).reshape(self.tr_N, self.H[hop], self.W[hop], -1)
                    tr_probmap_sibling = self.gather_neighbour(self.tr_prob[hop],hopidx=hop).reshape(self.tr_N, self.H[hop], self.W[hop], -1)
                    te_probmap_child = self.gather_children(self.te_prob[hop-1],hopidx=hop).reshape(self.te_N, self.H[hop], self.W[hop], -1)
                    te_probmap_sibling = self.gather_neighbour(self.te_prob[hop],hopidx=hop).reshape(self.te_N, self.H[hop], self.W[hop], -1)
                    if self.parent and (hop<self.numHop-1):# not the last hop, have parent
                        tr_probmap_parent = self.gather_parent(self.tr_prob[hop+1], hopidx=hop).reshape(self.tr_N, self.H[hop], self.W[hop], -1)
                        te_probmap_parent = self.gather_parent(self.te_prob[hop+1], hopidx=hop).reshape(self.te_N, self.H[hop], self.W[hop], -1)
                        tr_probmap_agg = np.concatenate((tr_probmap_parent, tr_probmap_sibling, tr_probmap_child),axis=-1)
                        te_probmap_agg = np.concatenate((te_probmap_parent, te_probmap_sibling, te_probmap_child),axis=-1)
                    else:
                        tr_probmap_agg = np.concatenate((tr_probmap_sibling, tr_probmap_child),axis=-1)
                        te_probmap_agg = np.concatenate((te_probmap_sibling, te_probmap_child),axis=-1)
                    
                # if self.average:
                #     tr_probmap_agg = np.mean(tr_probmap_agg,axis=-1)
                #     te_probmap_agg = np.mean(te_probmap_agg,axis=-1)
                    
                    
                if (self.saabfeat or ite<1):
                    if hop>0:
                        tr_X_ite = np.concatenate((tr_X[hop].reshape(self.tr_N, self.H[hop], self.W[hop], -1), tr_probmap_agg.reshape(self.tr_N, self.H[hop], self.W[hop], -1)),axis=-1)
                        te_X_ite = np.concatenate((te_X[hop].reshape(self.te_N, self.H[hop], self.W[hop], -1), te_probmap_agg.reshape(self.te_N, self.H[hop], self.W[hop], -1)),axis=-1)
                    else:
                        tr_X_ite = np.copy(tr_X[hop].reshape(self.tr_N, self.H[hop], self.W[hop], -1))
                        te_X_ite = np.copy(te_X[hop].reshape(self.te_N, self.H[hop], self.W[hop], -1))
                else:
                    tr_X[hop] = []
                    te_X[hop] = []
                    tr_X_ite = tr_probmap_agg.reshape(self.tr_N, self.H[hop], self.W[hop], -1)
                    te_X_ite = te_probmap_agg.reshape(self.te_N, self.H[hop], self.W[hop], -1)
                    
                    
                self.tr_prob[hop], self.te_prob[hop] = self.fit_inner(tr_X_ite, tr_Y[hop], te_X_ite, te_Y[hop], hopidx=hop, iteidx = ite)
                
                del tr_X_ite,te_X_ite

                self.tr_prob_save[hop].append(self.tr_prob[hop].reshape(self.tr_N, self.H[hop], self.W[hop], -1))
                self.te_prob_save[hop].append(self.te_prob[hop].reshape(self.te_N, self.H[hop], self.W[hop], -1))
                
        
    def fit_inner(self, tr_X_ite, tr_Y, te_X_ite, te_Y, hopidx=0,iteidx=0):
        if self.model_ready == False:
            if iteidx==0:
                if self.small_model:
                    MAX_DEPTH = 3
                    N_E = 100
                else:
                    MAX_DEPTH = 6
                    N_E = 300
            else:
                MAX_DEPTH = 6
                N_E = 300
                
            if self.gpu is None:
                clf = xgb.XGBClassifier(n_jobs=-1,
                                        objective='multi:softprob',
                                        # tree_method='gpu_hist', gpu_id=FLAGS.GPU,
                                        max_depth=MAX_DEPTH, n_estimators=N_E,
                                        min_child_weight=20,gamma=10,
                                        subsample=0.8,learning_rate=0.1,
                                        nthread=8,colsample_bytree=1.0).fit(tr_X_ite.reshape(self.tr_N*self.H[hopidx]*self.W[hopidx], -1), tr_Y.reshape(-1),
                                                                            early_stopping_rounds=100,eval_metric=self.eval,#['error','auc','logloss'],
                                                                            xgb_model=None, #,
                                                                            eval_set=[(te_X_ite.reshape(self.te_N*self.H[hopidx]*self.W[hopidx], -1), te_Y.reshape(-1))])        
            else:
                if iteidx==0:
                    clf = xgb.XGBClassifier(n_jobs=-1,
                                            objective='multi:softprob',
                                            tree_method='gpu_hist', gpu_id=self.gpu,
                                            max_depth=MAX_DEPTH, n_estimators=N_E,
                                            min_child_weight=20,gamma=10,
                                            subsample=0.8,learning_rate=0.1,
                                            nthread=8,colsample_bytree=1.0).fit(tr_X_ite[:50000].reshape(50000*self.H[hopidx]*self.W[hopidx], -1), tr_Y.reshape(-1)[:(50000*self.H[hopidx]*self.W[hopidx])],
                                                                                early_stopping_rounds=100,eval_metric=self.eval,#['error','auc','logloss'],
                                                                                xgb_model=None, #self.XGB_previous_[ite],
                                                                                eval_set=[(te_X_ite[:10000].reshape(10000*self.H[hopidx]*self.W[hopidx], -1), te_Y.reshape(-1)[:(10000*self.H[hopidx]*self.W[hopidx])])])        
                else:
                    clf = xgb.XGBClassifier(n_jobs=-1,
                                            objective='multi:softprob',
                                            tree_method='gpu_hist', gpu_id=self.gpu,
                                            max_depth=MAX_DEPTH, n_estimators=N_E,
                                            min_child_weight=20,gamma=10,
                                            subsample=0.8,learning_rate=0.1,
                                            nthread=8,colsample_bytree=1.0).fit(tr_X_ite.reshape(self.tr_N*self.H[hopidx]*self.W[hopidx], -1), tr_Y.reshape(-1),
                                                                                early_stopping_rounds=100,eval_metric=self.eval,#['error','auc','logloss'],
                                                                                xgb_model=None, #self.XGB_previous_[ite],
                                                                                # eval_set=[(te_X_ite[:10000].reshape(self.te_N*self.H[hopidx]*self.W[hopidx], -1), te_Y.reshape(-1))])        
                                                                                eval_set=[(te_X_ite.reshape(self.te_N*self.H[hopidx]*self.W[hopidx], -1), te_Y.reshape(-1))])        
                                                                            
            # clf = RFC(n_jobs=8, n_estimators=100, max_depth=30, oob_score=True).fit(tr_X_ite.reshape(self.tr_N*self.H[hopidx]*self.W[hopidx], -1), tr_Y.reshape(-1))
                                                                            
            # self.XGB_previous_.append(clf.get_booster())
            self.models['Hop{}_iter{}'.format(hopidx,iteidx)] = clf
            # self.xgbloss.append(clf.evals_result()['validation_0'])                                                        
            # self.NUM_ROUNDS.append(len(self.xgbloss[ite]['logloss']))
            self.feature_imp.append(clf.feature_importances_)
        else:
            clf = self.models['Hop{}_iter{}'.format(hopidx,iteidx)] 
        
        
        tr_prob = clf.predict_proba(tr_X_ite.reshape(self.tr_N*self.H[hopidx]*self.W[hopidx], -1))
        te_prob = clf.predict_proba(te_X_ite.reshape(self.te_N*self.H[hopidx]*self.W[hopidx], -1))
        print(tr_prob.shape)
        print(te_prob.shape)
        
        self.tr_loss[hopidx].append(self.evaluate(tr_prob, tr_Y.reshape(-1), eval_metric=self.eval))
        self.te_loss[hopidx].append(self.evaluate(te_prob, te_Y.reshape(-1), eval_metric=self.eval))
        self.tr_acc[hopidx].append(1-self.evaluate(tr_prob, tr_Y.reshape(-1), eval_metric='merror'))
        self.te_acc[hopidx].append(1-self.evaluate(te_prob, te_Y.reshape(-1), eval_metric='merror'))
        print('==========================================================')
        print('Hopindex #{}: tr_loss = {}'.format(hopidx, self.tr_loss[hopidx]))
        print('Hopindex #{}: te_loss = {}'.format(hopidx, self.te_loss[hopidx]))
        print('Hopindex #{}: tr_acc = {}'.format(hopidx, self.tr_acc[hopidx]))
        print('Hopindex #{}: te_acc = {}'.format(hopidx, self.te_acc[hopidx]))
        print('==========================================================')
        
        return tr_prob[:,1:], te_prob[:,1:]
        
        
            
            
    def predict_proba(self, feat_X, sing=0):
        proba = self.models[-1].predict_proba(feat_X)
        if sing==1:
            proba = proba[:,-1]
        return proba.squeeze()
    
    
    
    def evaluate(self, y_prob, gt, eval_metric):
        loss=0
        if eval_metric =='mlogloss':
            loss = log_loss(gt, y_prob)
        elif eval_metric=='merror':
            loss = np.sum(np.argmax(y_prob,axis=1)!=gt)/gt.size
        return loss
        
    
    def gather_neighbour(self, prob_map, win=3, hopidx=0):
        ## default is averaging
        prob_map = prob_map.reshape(-1, self.H[hopidx], self.W[hopidx], 9)
        if self.version<4:
            prob_map_new = view_as_windows(np.pad(prob_map,((0,0),(int(win//2),int(win//2)),
                                                            (int(win//2),int(win//2)),(0,0)),mode='reflect'),
                                           (1,win,win,1)).squeeze()
        else:
            if self.H[hopidx]>1:
                tmp = view_as_windows(np.pad(prob_map,((0,0),(int(win//2),int(win//2)),(int(win//2),int(win//2)),(0,0))), (1,win,win,1)).squeeze()
                print(tmp.shape)
                tmp = tmp.reshape(-1, self.H[hopidx], self.W[hopidx], 9, win*win)
                if self.neigh4:
                    tmp = tmp[:,:,:,1::2]
                    non_zeros_num = np.count_nonzero(tmp, axis=-1).reshape(-1, self.H[hopidx], self.W[hopidx], 9)
                    tmp = np.sum(tmp,axis=-1)
                else:## neigh8
                    non_zeros_num = np.count_nonzero(tmp, axis=-1).reshape(-1, self.H[hopidx], self.W[hopidx], 9) - 1
                    tmp = np.sum(tmp,axis=-1)
                    tmp = tmp - prob_map
                    
                prob_map_new = tmp/non_zeros_num # already the mean
                del tmp
                
        if self.H[hopidx]>1: 
            print(prob_map_new.shape)
            prob_map_new = prob_map_new.reshape(prob_map.shape[0], self.H[hopidx], self.W[hopidx], 9)
            # prob_map_new = np.moveaxis(prob_map_new,3,-1)
            if self.rank_neighbour:
                print('Rank')
                prob_map_new = np.sort(prob_map_new, axis=-1)
                # prob_map_new = np.concatenate((prob_map_new,1-prob_map_new),axis=-1)
            prob_map_new = np.concatenate((prob_map, prob_map_new),axis=-1) # concatenate itself
        else:
            prob_map_new = np.copy(prob_map)
        
        
        return prob_map_new
        
    def gather_children(self, prob_map, hopidx=1):
        if self.version<5:
            win = 3
            stride=1
        elif self.version==5: ## version==5
            if hopidx==1:
                win = 7
                stride = 2
            else:
                win = 3
                stride = 1
        else:## version>5
            win = 3
            stride = 1
                
        prob_map = prob_map.reshape(-1, self.H[hopidx-1], self.W[hopidx-1], 9)
        
        if self.version<4:
            prob_map_new = view_as_windows(np.pad(prob_map,((0,0),(1,1),(1,1),(0,0)),mode='reflect'),
                                           (1,win,win,1),(1,stride,stride,1)).squeeze()
        else:
            prob_map_new = view_as_windows(prob_map,(1,win,win,1),(1,stride,stride,1)).squeeze()
            
        print(prob_map_new.shape)
        prob_map_new = prob_map_new.reshape(-1, self.H[hopidx], self.W[hopidx], 9, win*win)

        if self.rank_neighbour:
            print('Rank')
            prob_map_new = np.sort(prob_map_new, axis=-1)
            # prob_map_new = np.concatenate((prob_map_new,1-prob_map_new),axis=-1)
            
        if self.average:
            prob_map_new = np.mean(prob_map_new,axis=-1)
        else:
            prob_map_new = prob_map_new.reshape(-1, self.H[hopidx], self.W[hopidx], 9, win*win)
        
        return prob_map_new
        
        
    def gather_parent(self, prob_map_ori, hopidx=0):
        prob_map = prob_map_ori.reshape(-1, self.H[hopidx+1], self.W[hopidx+1], 9)
        print('parent ori shape={}'.format(prob_map.shape))
        if self.version==1:
            if hopidx==0:
                prob_map_new = cv2.resize(prob_map.squeeze(),(1,13,13),cv2.INTERAREA)
                prob_map_new = np.pad(prob_map_new,((0,0),(1,1),(1,1),(0,0)),mode='edge')
                agg=1
            else:
                prob_map_new = np.copy(prob_map)
                agg=1
                
        elif self.version==2:
            if hopidx==0:
                prob_map_new = np.kron(prob_map, np.ones((1,2,2,1)))
                agg=1
            else:
                prob_map_new = np.copy(prob_map)
                agg=1
        elif self.version==3:
            prob_map_new = np.copy(prob_map)
            agg=1
        elif self.version==4:
            prob_map_new = np.pad(prob_map,((0,0),(1,1),(1,1),(0,0)),mode='edge')
            agg=1
        elif self.version==5:
            if hopidx==0:
                prob_map_new = np.zeros((prob_map.shape[0],9,9,9))
                for n in range(prob_map.shape[0]):
                    prob_map_new[n,:,:,:]= cv2.resize(prob_map[n].squeeze(),(9,9),interpolation=cv2.INTER_AREA)
                prob_map_new = np.pad(prob_map_new,((0,0),(3,3),(3,3),(0,0)),mode='edge')
            else:
                prob_map_new = np.pad(prob_map,((0,0),(1,1),(1,1),(0,0)),mode='edge')
            agg=9
        elif self.version==6:
            prob_map_new = np.pad(prob_map,((0,0),(1,1),(1,1),(0,0)),mode='edge')
            agg = 9
            
        print('parent shape={}'.format(prob_map_new.shape))
        prob_map_new = prob_map_new.reshape(-1, self.H[hopidx], self.W[hopidx], agg)
        print('parent shape2={}'.format(prob_map_new.shape))
        
        if self.rank_neighbour:
            print('Rank')
            prob_map_new = np.sort(prob_map_new, axis=-1)
            # prob_map_new = np.concatenate((prob_map_new,1-prob_map_new),axis=-1)
        
        # if self.average:
        #     prob_map_new = np.mean(prob_map_new,axis=-1)
        
        return prob_map_new
        
        
        
        
        
        
        
        
        
    
    
      