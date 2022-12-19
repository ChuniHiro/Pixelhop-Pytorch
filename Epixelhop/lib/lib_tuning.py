from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression



def tune_LR(train_flat, train_labels,params=None,folds=5,param_comb=10):
    if params is None:
        params = {
                'C': [0.001,0.005,0.01,0.05,0.1,1,10]
                }
        
    clf = LogisticRegression(solver='saga', class_weight='balanced')
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
    random_search = GridSearchCV(clf, param_grid=params, scoring='roc_auc', 
                                       n_jobs=8, cv=skf.split(train_flat,train_labels), verbose=10)
    # # Here we go
    random_search.fit(train_flat, train_labels)
    # train_preds = random_search.best_estimator_.predict_proba(train_flat)
    print(random_search.best_estimator_)
    return random_search.best_estimator_    

def tune_xgb(train_flat, train_labels,GPU=None,params=None,folds=5,param_comb=10):
    if params is None:
        params = {
                'objective': ['binary:logistic'],
                'n_estimators': [200,300,500],
                'max_depth': [3,4,5,6],
                'min_child_weight': [4, 5, 10, 15, 21],
                'gamma': [2, 5,10,15,20],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'learning_rate': [0.08, 0.1]#,
                #'early_stopping_rounds': [30,40,50]
                }
        
        
    xgb = XGBClassifier(eval_metric='auc')#,tree_method='gpu_hist', gpu_id=GPU)
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', 
                                       n_jobs=8, cv=skf.split(train_flat,train_labels), verbose=10)#, random_state=1001)
    
    # # Here we go
    random_search.fit(train_flat, train_labels)
    # train_preds = random_search.best_estimator_.predict_proba(train_flat)
    print(random_search.best_estimator_)
    return random_search.best_estimator_


def tune_xgb_grid(train_flat, train_labels,params=None,folds=5,param_comb=10):
    if params is None:
        params = {
                # 'objective': ['binary:logistic'],
                'n_estimators': [100,200],
                'max_depth': [3,4,5,6],
                'min_child_weight': [4, 5, 10, 15, 21],
                'gamma': [2, 5,10,15,20],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'learning_rate': [0.08, 0.1],
                'early_stopping_rounds': [30,40,50]
                }
        
        
    xgb = XGBClassifier(eval_metric='auc')
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
    random_search = GridSearchCV(xgb, param_grid=params, scoring='roc_auc', 
                                       n_jobs=8, cv=skf.split(train_flat,train_labels), verbose=10)
    
    # # Here we go
    random_search.fit(train_flat, train_labels)
    # train_preds = random_search.best_estimator_.predict_proba(train_flat)
    print(random_search.best_estimator_)
    return random_search.best_estimator_