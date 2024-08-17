#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression


# ## Load source datasets

# In[2]:


train_df = pd.read_csv("../input/tabular-playground-series-sep-2021/train.csv")
train_df.set_index('id', inplace=True)
print(f"train_df: {train_df.shape}")
train_df.head()


# In[3]:


test_df = pd.read_csv("../input/tabular-playground-series-sep-2021/test.csv")
test_df.set_index('id', inplace=True)
print(f"test_df: {test_df.shape}")
test_df.head()


# ## Feature Engineering

# In[4]:


features = test_df.columns.tolist()

train_df['num_missing'] = train_df[features].isna().sum(axis=1)
train_df['num_missing_std'] = train_df[features].isna().std(axis=1).astype('float')
train_df['median'] = train_df[features].median(axis=1)
train_df['std'] = train_df[features].std(axis=1)
train_df['min'] = train_df[features].abs().min(axis=1)
train_df['max'] = train_df[features].abs().max(axis=1)
train_df['sem'] = train_df[features].sem(axis=1)

test_df['num_missing'] = test_df[features].isna().sum(axis=1)
test_df['num_missing_std'] = test_df[features].isna().std(axis=1).astype('float')
test_df['median'] = test_df[features].median(axis=1)
test_df['std'] = test_df[features].std(axis=1)
test_df['min'] = test_df[features].abs().min(axis=1)
test_df['max'] = test_df[features].abs().max(axis=1)
test_df['sem'] = test_df[features].sem(axis=1)

print(f"train_df: {train_df.shape} \ntest_df: {test_df.shape}")
train_df.head()


# In[5]:


fill_value_dict = {
    'f1': 'Mean', 
    'f2': 'Mean', 
    'f3': 'Mode', 
    'f4': 'Mode', 
    'f5': 'Mode', 
    'f6': 'Mean', 
    'f7': 'Mean', 
    'f8': 'Median', 
    'f9': 'Mode', 
    'f10': 'Mode', 
    'f11': 'Mode', 
    'f12': 'Median', 
    'f13': 'Mode', 
    'f14': 'Median', 
    'f15': 'Mean', 
    'f16': 'Median', 
    'f17': 'Mode', 
    'f18': 'Median', 
    'f19': 'Median', 
    'f20': 'Median', 
    'f21': 'Median', 
    'f22': 'Mean', 
    'f23': 'Mode', 
    'f24': 'Median', 
    'f25': 'Median', 
    'f26': 'Median', 
    'f27': 'Median', 
    'f28': 'Median', 
    'f29': 'Mean', 
    'f30': 'Median', 
    'f31': 'Mode', 
    'f32': 'Median', 
    'f33': 'Median', 
    'f34': 'Mean', 
    'f35': 'Median', 
    'f36': 'Median', 
    'f37': 'Median', 
    'f38': 'Mode', 
    'f39': 'Median', 
    'f40': 'Mean', 
    'f41': 'Median', 
    'f42': 'Mean', 
    'f43': 'Mode', 
    'f44': 'Median', 
    'f45': 'Median', 
    'f46': 'Mean', 
    'f47': 'Mean', 
    'f48': 'Median', 
    'f49': 'Mode', 
    'f50': 'Mean', 
    'f51': 'Median', 
    'f52': 'Median', 
    'f53': 'Median', 
    'f54': 'Median', 
    'f55': 'Mode', 
    'f56': 'Mean', 
    'f57': 'Mean', 
    'f58': 'Median', 
    'f59': 'Median', 
    'f60': 'Mode', 
    'f61': 'Mode', 
    'f62': 'Median', 
    'f63': 'Median', 
    'f64': 'Median', 
    'f65': 'Mean', 
    'f66': 'Mode', 
    'f67': 'Median', 
    'f68': 'Median', 
    'f69': 'Mode', 
    'f70': 'Mean', 
    'f71': 'Median', 
    'f72': 'Median', 
    'f73': 'Median', 
    'f74': 'Median', 
    'f75': 'Mean', 
    'f76': 'Mean', 
    'f77': 'Median', 
    'f78': 'Median', 
    'f79': 'Median', 
    'f80': 'Median', 
    'f81': 'Median', 
    'f82': 'Median', 
    'f83': 'Median', 
    'f84': 'Median', 
    'f85': 'Median', 
    'f86': 'Median', 
    'f87': 'Median', 
    'f88': 'Median', 
    'f89': 'Median', 
    'f90': 'Mean', 
    'f91': 'Mode', 
    'f92': 'Median', 
    'f93': 'Median', 
    'f94': 'Mode', 
    'f95': 'Median', 
    'f96': 'Median', 
    'f97': 'Mean', 
    'f98': 'Median', 
    'f99': 'Median', 
    'f100': 'Mean', 
    'f101': 'Median', 
    'f102': 'Median', 
    'f103': 'Median', 
    'f104': 'Median', 
    'f105': 'Mode', 
    'f106': 'Median', 
    'f107': 'Median', 
    'f108': 'Median', 
    'f109': 'Median', 
    'f110': 'Mode', 
    'f111': 'Median', 
    'f112': 'Median', 
    'f113': 'Median', 
    'f114': 'Median', 
    'f115': 'Mode', 
    'f116': 'Median', 
    'f117': 'Median', 
    'f118': 'Mean'
}


for col in tqdm(features):
    if fill_value_dict.get(col)=='Mean':
        fill_value = train_df[col].mean()
    elif fill_value_dict.get(col)=='Median':
        fill_value = train_df[col].median()
    elif fill_value_dict.get(col)=='Mode':
        fill_value = train_df[col].mode().iloc[0]
    
    train_df[col].fillna(fill_value, inplace=True)
    test_df[col].fillna(fill_value, inplace=True)

train_df.head()


# In[6]:


features = [col for col in train_df.columns if col not in ['num_missing','num_missing_std','claim']]

for col in tqdm(features):
    transformer = QuantileTransformer(n_quantiles=3000, 
                                      random_state=42, 
                                      output_distribution="normal")
    
    vec_len = len(train_df[col].values)
    vec_len_test = len(test_df[col].values)

    raw_vec = train_df[col].values.reshape(vec_len, 1)
    test_vec = test_df[col].values.reshape(vec_len_test, 1)
    transformer.fit(raw_vec)
    
    train_df[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test_df[col] = transformer.transform(test_vec).reshape(1, vec_len_test)[0]

print(f"train_df: {train_df.shape} \ntest_df: {test_df.shape}")


# In[7]:


def kmeans_fet(train, test, features, n_clusters):
    
    train_ = train[features].copy()
    test_ = test[features].copy()
    data = pd.concat([train_, test_], axis=0)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
    
    train[f'clusters_k'] = kmeans.labels_[:train.shape[0]]
    test[f'clusters_k'] = kmeans.labels_[train.shape[0]:]
    return train, test


# In[8]:


train_df, test_df = kmeans_fet(train_df, test_df, features, n_clusters=4)

Xtrain = train_df.loc[:, train_df.columns != 'claim'].copy()
Ytrain = train_df['claim'].copy()
Xtest = test_df.copy()

print(f"Xtrain: {Xtrain.shape} \nYtrain: {Ytrain.shape} \nXtest: {Xtest.shape}")

del train_df
del test_df
gc.collect()


# ## Model Hyperparameters

# In[9]:


FOLD = 5
SEEDS = [24, 42]


lgb_params1 = {
    'metric' : 'auc',
    'objective' : 'binary',
    'device_type': 'gpu', 
    'n_estimators': 10000, 
    'learning_rate': 0.12230165751633416, 
    'num_leaves': 1400, 
    'max_depth': 8, 
    'min_child_samples': 300, 
    'reg_alpha': 10, 
    'reg_lambda': 65, 
    'min_split_gain': 5.157818977461183, 
    'subsample': 0.5, 
    'subsample_freq': 1, 
    'colsample_bytree': 0.2,
    'random_state': 42
}

lgb_params2 = {
    'metric' : 'auc',
    'max_depth' : 3,
    'num_leaves' : 7,
    'n_estimators' : 5000,
    'colsample_bytree' : 0.3,
    'subsample' : 0.5,
    'random_state' : 42,
    'reg_alpha' : 18,
    'reg_lambda' : 17,
    'learning_rate' : 0.095,
    'device' : 'gpu',
    'objective' : 'binary'
}

cb_params1 = {
    'eval_metric' : 'AUC',
    'iterations': 15585, 
    'objective': 'CrossEntropy',
    'bootstrap_type': 'Bernoulli', 
    'od_wait': 1144, 
    'learning_rate': 0.023575206684596582, 
    'reg_lambda': 36.30433203563295, 
    'random_strength': 43.75597655616195, 
    'depth': 7, 
    'min_data_in_leaf': 11, 
    'leaf_estimation_iterations': 1, 
    'subsample': 0.8227911142845009,
    'task_type' : 'GPU',
    'devices' : '0',
    'verbose' : 0,
    'random_state': 42
}

cb_params2 = {
    'eval_metric' : 'AUC',
    'depth' : 5,
    'grow_policy' : 'SymmetricTree',
    'l2_leaf_reg' : 3.0,
    'random_strength' : 1.0,
    'learning_rate' : 0.1,
    'iterations' : 10000,
    'loss_function' : 'CrossEntropy',
    'task_type' : 'GPU',
    'devices' : '0',
    'verbose' : 0,
    'random_state': 42
}

xgb_params1 = {
    'eval_metric': 'auc', 
    'objective': 'binary:logistic', 
    'tree_method': 'gpu_hist', 
    'gpu_id': 0, 
    'predictor': 'gpu_predictor', 
    'n_estimators': 10000, 
    'learning_rate': 0.01063045229441343, 
    'gamma': 0.24652519525750877, 
    'max_depth': 4, 
    'min_child_weight': 366, 
    'subsample': 0.6423040816299684, 
    'colsample_bytree': 0.7751264493218339, 
    'colsample_bylevel': 0.8675692743597421, 
    'lambda': 0, 
    'alpha': 10,
    'random_state': 42
}

xgb_params2 = {
    'eval_metric': 'auc',
    'max_depth': 3,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'learning_rate': 0.01187431306013263,
    'n_estimators': 10000,
    'n_jobs': -1,
    'use_label_encoder': False,
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor',
    'random_state': 42
}


# ## XGBoost Model

# In[10]:


counter = 0
oof_score = 0
y_pred_final_xgb1 = np.zeros((Xtest.shape[0], 1))
y_pred_meta_xgb1 = np.zeros((Xtrain.shape[0], 1))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train, val) in enumerate(kfold.split(Xtrain.values, Ytrain.values)):
        counter += 1

        train_x, train_y = Xtrain.iloc[train], Ytrain.iloc[train]
        val_x, val_y = Xtrain.iloc[val], Ytrain.iloc[val]

        model = XGBClassifier(**xgb_params1)

        model.fit(train_x, train_y, eval_set=[(train_x, train_y), (val_x, val_y)], 
                  early_stopping_rounds=200, verbose=1000)
        
        y_pred = model.predict_proba(val_x, iteration_range=(0, model.best_iteration))[:,-1]
        y_pred_meta_xgb1[val] += np.array([y_pred]).T
        y_pred_final_xgb1 += np.array([model.predict_proba(Xtest, iteration_range=(0, model.best_iteration))[:,-1]]).T
        
        score = roc_auc_score(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("\nXGBoost | Seed-{} | Fold-{} | OOF Score: {}\n".format(seed, idx, score))
        
        del model, y_pred
        del train_x, train_y
        del val_x, val_y
        gc.collect()
    
    print("\nXGBoost | Seed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_meta_xgb1 = y_pred_meta_xgb1 / float(len(SEEDS))
y_pred_final_xgb1 = y_pred_final_xgb1 / float(counter)
oof_score /= float(counter)
print("XGBoost | Aggregate OOF Score: {}".format(oof_score))


# In[11]:


counter = 0
oof_score = 0
y_pred_final_xgb2 = np.zeros((Xtest.shape[0], 1))
y_pred_meta_xgb2 = np.zeros((Xtrain.shape[0], 1))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train, val) in enumerate(kfold.split(Xtrain.values, Ytrain.values)):
        counter += 1

        train_x, train_y = Xtrain.iloc[train], Ytrain.iloc[train]
        val_x, val_y = Xtrain.iloc[val], Ytrain.iloc[val]

        model = XGBClassifier(**xgb_params2)

        model.fit(train_x, train_y, eval_set=[(train_x, train_y), (val_x, val_y)], 
                  early_stopping_rounds=200, verbose=1000)
        
        y_pred = model.predict_proba(val_x, iteration_range=(0, model.best_iteration))[:,-1]
        y_pred_meta_xgb2[val] += np.array([y_pred]).T
        y_pred_final_xgb2 += np.array([model.predict_proba(Xtest, iteration_range=(0, model.best_iteration))[:,-1]]).T
        
        score = roc_auc_score(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("\nXGBoost | Seed-{} | Fold-{} | OOF Score: {}\n".format(seed, idx, score))
        
        del model, y_pred
        del train_x, train_y
        del val_x, val_y
        gc.collect()
    
    print("\nXGBoost | Seed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_meta_xgb2 = y_pred_meta_xgb2 / float(len(SEEDS))
y_pred_final_xgb2 = y_pred_final_xgb2 / float(counter)
oof_score /= float(counter)
print("XGBoost | Aggregate OOF Score: {}".format(oof_score))


# In[12]:


np.savez_compressed('./XGB_Meta_Features.npz',
                    y_pred_meta_xgb1=y_pred_meta_xgb1, 
                    y_pred_meta_xgb2=y_pred_meta_xgb2, 
                    y_pred_final_xgb1=y_pred_final_xgb1,
                    y_pred_final_xgb2=y_pred_final_xgb2)


# ## LightGBM Model

# In[13]:


counter = 0
oof_score = 0
y_pred_final_lgb1 = np.zeros((Xtest.shape[0], 1))
y_pred_meta_lgb1 = np.zeros((Xtrain.shape[0], 1))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train, val) in enumerate(kfold.split(Xtrain, Ytrain)):
        counter += 1

        train_x, train_y = Xtrain.iloc[train], Ytrain.iloc[train]
        val_x, val_y = Xtrain.iloc[val], Ytrain.iloc[val]

        model = LGBMClassifier(**lgb_params1)
        
        model.fit(train_x, train_y, eval_metric='auc',
                  eval_set=[(train_x, train_y), (val_x, val_y)],
                  early_stopping_rounds=200, verbose=500)

        y_pred = model.predict_proba(val_x, num_iteration=model.best_iteration_)[:,-1]
        y_pred_meta_lgb1[val] += np.array([y_pred]).T
        y_pred_final_lgb1 += np.array([model.predict_proba(Xtest, num_iteration=model.best_iteration_)[:,-1]]).T
        
        score = roc_auc_score(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("\nLightGBM | Seed-{} | Fold-{} | OOF Score: {}\n".format(seed, idx, score))
        
        del model, y_pred
        del train_x, train_y
        del val_x, val_y
        gc.collect()
    
    print("\nLightGBM | Seed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_meta_lgb1 = y_pred_meta_lgb1 / float(len(SEEDS))
y_pred_final_lgb1 = y_pred_final_lgb1 / float(counter)
oof_score /= float(counter)
print("LightGBM | Aggregate OOF Score: {}".format(oof_score))


# In[14]:


counter = 0
oof_score = 0
y_pred_final_lgb2 = np.zeros((Xtest.shape[0], 1))
y_pred_meta_lgb2 = np.zeros((Xtrain.shape[0], 1))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train, val) in enumerate(kfold.split(Xtrain, Ytrain)):
        counter += 1

        train_x, train_y = Xtrain.iloc[train], Ytrain.iloc[train]
        val_x, val_y = Xtrain.iloc[val], Ytrain.iloc[val]

        model = LGBMClassifier(**lgb_params2)
        
        model.fit(train_x, train_y, eval_metric='auc',
                  eval_set=[(train_x, train_y), (val_x, val_y)],
                  early_stopping_rounds=200, verbose=500)

        y_pred = model.predict_proba(val_x, num_iteration=model.best_iteration_)[:,-1]
        y_pred_meta_lgb2[val] += np.array([y_pred]).T
        y_pred_final_lgb2 += np.array([model.predict_proba(Xtest, num_iteration=model.best_iteration_)[:,-1]]).T
        
        score = roc_auc_score(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("\nLightGBM | Seed-{} | Fold-{} | OOF Score: {}\n".format(seed, idx, score))
        
        del model, y_pred
        del train_x, train_y
        del val_x, val_y
        gc.collect()
    
    print("\nLightGBM | Seed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_meta_lgb2 = y_pred_meta_lgb2 / float(len(SEEDS))
y_pred_final_lgb2 = y_pred_final_lgb2 / float(counter)
oof_score /= float(counter)
print("LightGBM | Aggregate OOF Score: {}".format(oof_score))


# In[15]:


np.savez_compressed('./LGB_Meta_Features.npz',
                    y_pred_meta_lgb1=y_pred_meta_lgb1, 
                    y_pred_meta_lgb2=y_pred_meta_lgb2, 
                    y_pred_final_lgb1=y_pred_final_lgb1,
                    y_pred_final_lgb2=y_pred_final_lgb2)


# ## CatBoost Model

# In[16]:


counter = 0
oof_score = 0
y_pred_final_cb1 = np.zeros((Xtest.shape[0], 1))
y_pred_meta_cb1 = np.zeros((Xtrain.shape[0], 1))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train, val) in enumerate(kfold.split(Xtrain.values, Ytrain.values)):
        counter += 1

        train_x, train_y = Xtrain.iloc[train], Ytrain.iloc[train]
        val_x, val_y = Xtrain.iloc[val], Ytrain.iloc[val]

        model = CatBoostClassifier(**cb_params1)

        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], 
                  early_stopping_rounds=200, verbose=1000)

        y_pred = model.predict_proba(val_x)[:,-1]
        y_pred_meta_cb1[val] += np.array([y_pred]).T
        y_pred_final_cb1 += np.array([model.predict_proba(Xtest)[:,-1]]).T
        
        score = roc_auc_score(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("\nCatBoost | Seed-{} | Fold-{} | OOF Score: {}\n".format(seed, idx, score))
        
        del model, y_pred
        del train_x, train_y
        del val_x, val_y
        gc.collect()
    
    print("\nCatBoost | Seed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_meta_cb1 = y_pred_meta_cb1 / float(len(SEEDS))
y_pred_final_cb1 = y_pred_final_cb1 / float(counter)
oof_score /= float(counter)
print("CatBoost | Aggregate OOF Score: {}".format(oof_score))


# In[17]:


counter = 0
oof_score = 0
y_pred_final_cb2 = np.zeros((Xtest.shape[0], 1))
y_pred_meta_cb2 = np.zeros((Xtrain.shape[0], 1))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train, val) in enumerate(kfold.split(Xtrain.values, Ytrain.values)):
        counter += 1

        train_x, train_y = Xtrain.iloc[train], Ytrain.iloc[train]
        val_x, val_y = Xtrain.iloc[val], Ytrain.iloc[val]

        model = CatBoostClassifier(**cb_params1)

        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], 
                  early_stopping_rounds=200, verbose=500)

        y_pred = model.predict_proba(val_x)[:,-1]
        y_pred_meta_cb2[val] += np.array([y_pred]).T
        y_pred_final_cb2 += np.array([model.predict_proba(Xtest)[:,-1]]).T
        
        score = roc_auc_score(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("\nCatBoost | Seed-{} | Fold-{} | OOF Score: {}\n".format(seed, idx, score))
        
        del model, y_pred
        del train_x, train_y
        del val_x, val_y
        gc.collect()
    
    print("\nCatBoost | Seed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_meta_cb2 = y_pred_meta_cb2 / float(len(SEEDS))
y_pred_final_cb2 = y_pred_final_cb2 / float(counter)
oof_score /= float(counter)
print("CatBoost | Aggregate OOF Score: {}".format(oof_score))


# In[18]:


np.savez_compressed('./CB_Meta_Features.npz',
                    y_pred_meta_cb1=y_pred_meta_cb1, 
                    y_pred_meta_cb2=y_pred_meta_cb2, 
                    y_pred_final_cb1=y_pred_final_cb1,
                    y_pred_final_cb2=y_pred_final_cb2)


# ## Logistic Regression (Meta)

# In[19]:


Xtrain_meta = np.concatenate((y_pred_meta_cb1, y_pred_meta_lgb1, y_pred_meta_xgb1,
                              y_pred_meta_cb2, y_pred_meta_lgb2, y_pred_meta_xgb2), axis=1)
Xtest_meta = np.concatenate((y_pred_final_cb1, y_pred_final_lgb1, y_pred_final_xgb1,
                             y_pred_final_cb2, y_pred_final_lgb2, y_pred_final_xgb2), axis=1)

print("Xtrain_meta shape: {}".format(Xtrain_meta.shape))
print("Xtest_meta shape: {}".format(Xtest_meta.shape))

del Xtrain, Xtest
del y_pred_meta_cb1, y_pred_meta_lgb1, y_pred_meta_xgb1
del y_pred_meta_cb2, y_pred_meta_lgb2, y_pred_meta_xgb2
gc.collect()


# In[20]:


counter = 0
oof_score = 0
y_pred_final_lr = np.zeros((Xtest_meta.shape[0], 1))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train, val) in enumerate(kfold.split(Xtrain_meta, Ytrain)):
        counter += 1

        train_x, train_y = Xtrain_meta[train], Ytrain.iloc[train]
        val_x, val_y = Xtrain_meta[val], Ytrain.iloc[val]
        
        model = LogisticRegression(
            solver='saga',
            max_iter=1000,
            random_state=42
        )

        model.fit(train_x, train_y)

        y_pred = model.predict_proba(val_x)[:,-1]
        y_pred_final_lr += np.array([model.predict_proba(Xtest_meta)[:,-1]]).T
        
        score = roc_auc_score(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("LR | Seed-{} | Fold-{} | OOF Score: {}".format(seed, idx, score))
    
    print("\nLR | Seed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_final_lr = y_pred_final_lr / float(counter)
oof_score /= float(counter)
print("LR | Aggregate OOF Score: {}".format(oof_score))


# ## Create submission files

# In[21]:


submit_df = pd.read_csv("../input/tabular-playground-series-sep-2021/sample_solution.csv")
submit_df['claim'] = y_pred_final_lr.ravel()
submit_df.to_csv("Stacked_Submission.csv", index=False)
submit_df.head(10)


# In[22]:


y_pred_final_lgb = (y_pred_final_lgb1 + y_pred_final_lgb2)/2
y_pred_final_xgb = (y_pred_final_xgb1 + y_pred_final_xgb2)/2
y_pred_final_cb = (y_pred_final_cb1 + y_pred_final_cb2)/2

submit_df['claim'] = (y_pred_final_lr**4 + y_pred_final_xgb**4 + y_pred_final_lgb**4)/3
submit_df.to_csv("Power_Average_Submission.csv", index=False)
submit_df.head(10)


# In[23]:


submit_df['claim'] = (y_pred_final_lr * 0.55) + (y_pred_final_xgb * 0.25) + (y_pred_final_lgb * 0.15) + (y_pred_final_cb * 0.05)
submit_df.to_csv("Weighted_Average_Submission.csv", index=False)
submit_df.head(10)


# In[ ]:




