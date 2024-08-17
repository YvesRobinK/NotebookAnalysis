#!/usr/bin/env python
# coding: utf-8

# # Permutation Feature Importance

# I found the discussion post on feature importance by @ambrosm very interesting. So I wanted to try to implement permutation feature importance. 
# 
# The code below is derived from two notebooks by @cdeotte and I am using the dataset put together by @raddar. All links below
# 
# 1. https://www.kaggle.com/competitions/amex-default-prediction/discussion/331131
# 2. https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793
# 3. https://www.kaggle.com/code/cdeotte/lstm-feature-importance
# 4. https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format
# 
# I truly appreciate their work, please check it out! :)
# 
# 

# # Load libraries

# In[1]:


# LOAD LIBRARIES
import pandas as pd, numpy as np # CPU libraries
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import cupy, cudf # GPU libraries
import matplotlib.pyplot as plt, gc, os
import seaborn as sns
from tqdm.notebook import tqdm

print('RAPIDS version',cudf.__version__)


# In[2]:


# VERSION NAME FOR SAVED MODEL FILES
VER = 1

# TRAIN RANDOM SEED
SEED = 42

# FILL NAN VALUE
NAN_VALUE = -127 # will fit in int8

# FOLDS PER MODEL
FOLDS = 5


# # Process train data

# In[3]:


def read_file(path = '', usecols = None):
    # LOAD DATAFRAME
    if usecols is not None: df = cudf.read_parquet(path, columns=usecols)
    else: df = cudf.read_parquet(path)
    # REDUCE DTYPE FOR CUSTOMER AND DATE
    df['customer_ID'] = df['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    df.S_2 = cudf.to_datetime( df.S_2 )
    # SORT BY CUSTOMER AND DATE (so agg('last') works correctly)
    #df = df.sort_values(['customer_ID','S_2'])
    #df = df.reset_index(drop=True)
    # FILL NAN
    df = df.fillna(NAN_VALUE) 
    print('shape of data:', df.shape)
    
    return df

print('Reading train data...')
TRAIN_PATH = '../input/amex-data-integer-dtypes-parquet-format/train.parquet'
train = read_file(path = TRAIN_PATH)


# In[4]:


def process_and_feature_engineer(df):
    # FEATURE ENGINEERING FROM 
    # https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created
    all_cols = [c for c in list(df.columns) if c not in ['customer_ID','S_2']]
    cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
    num_features = [col for col in all_cols if col not in cat_features]

    test_num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]

    test_cat_agg = df.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]

    df = cudf.concat([test_num_agg, test_cat_agg], axis=1)
    del test_num_agg, test_cat_agg
    print('shape after engineering', df.shape )
    
    return df

train = process_and_feature_engineer(train)


# In[5]:


# ADD TARGETS
targets = cudf.read_csv('../input/amex-default-prediction/train_labels.csv')
targets['customer_ID'] = targets['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
targets = targets.set_index('customer_ID')
train = train.merge(targets, left_index=True, right_index=True, how='left')
train.target = train.target.astype('int8')
del targets

# NEEDED TO MAKE CV DETERMINISTIC (cudf merge above randomly shuffles rows)
train = train.sort_index().reset_index()

# FEATURES
FEATURES = train.columns[1:-1]
print(f'There are {len(FEATURES)} features!')


# # Setup for XGB Training

# In[6]:


# LOAD XGB LIBRARY
from sklearn.model_selection import KFold
import xgboost as xgb
print('XGB Version',xgb.__version__)

# XGB MODEL PARAMETERS
xgb_parms = { 
    'max_depth':4, 
    'learning_rate':0.05, 
    'subsample':0.8,
    'colsample_bytree':0.6, 
    'eval_metric':'logloss',
    'objective':'binary:logistic',
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'random_state':SEED
}


# In[7]:


# NEEDED WITH DeviceQuantileDMatrix BELOW
class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256*1024):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0 # set iterator to 0
        self.batch_size = batch_size
        self.batches = int( np.ceil( len(df) / self.batch_size ) )
        super().__init__()

    def reset(self):
        '''Reset the iterator'''
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data.'''
        if self.it == self.batches:
            return 0 # Return 0 when there's no more batch.
        
        a = self.it * self.batch_size
        b = min( (self.it + 1) * self.batch_size, len(self.df) )
        dt = cudf.DataFrame(self.df.iloc[a:b])
        input_data(data=dt[self.features], label=dt[self.target]) #, weight=dt['weight'])
        self.it += 1
        return 1


# In[8]:


# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)


# # Train XGB for single fold; Compute Permutation importance

# In[9]:


COMPUTE_PERM_IMPORTANCE = True
ONE_FOLD_ONLY = True


# In[10]:


importances = []
oof = []
train = train.to_pandas() # free GPU memory
TRAIN_SUBSAMPLE = 1.0
gc.collect()

skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold,(train_idx, valid_idx) in enumerate(skf.split(
            train, train.target )):
    
    # TRAIN WITH SUBSAMPLE OF TRAIN FOLD DATA
    if TRAIN_SUBSAMPLE<1.0:
        np.random.seed(SEED)
        train_idx = np.random.choice(train_idx, 
                       int(len(train_idx)*TRAIN_SUBSAMPLE), replace=False)
        np.random.seed(None)
    
    print('#'*25)
    print('### Fold',fold+1)
    print('### Train size',len(train_idx),'Valid size',len(valid_idx))
    print(f'### Training with {int(TRAIN_SUBSAMPLE*100)}% fold data...')
    print('#'*25)
    
    # TRAIN, VALID, TEST FOR FOLD K
    Xy_train = IterLoadForDMatrix(train.loc[train_idx], FEATURES, 'target')
    X_valid = train.loc[valid_idx, FEATURES]
    y_valid = train.loc[valid_idx, 'target']
    
    dtrain = xgb.DeviceQuantileDMatrix(Xy_train, max_bin=256)
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
    
    # TRAIN MODEL FOLD K
    model = xgb.train(xgb_parms, 
                dtrain=dtrain,
                evals=[(dtrain,'train'),(dvalid,'valid')],
                num_boost_round=9999,
                early_stopping_rounds=100,
                verbose_eval=100)
    model.save_model(f'XGB_v{VER}_fold{fold}.xgb')
    
    # GET FEATURE IMPORTANCE FOR FOLD K
    dd = model.get_score(importance_type='weight')
    df = pd.DataFrame({'feature':dd.keys(),f'importance_{fold}':dd.values()})
    importances.append(df)
            
    # INFER OOF FOLD K
    oof_preds = model.predict(dvalid)
    acc = amex_metric_mod(y_valid.values, oof_preds)
    print('Kaggle Metric =',acc,'\n')
    
    # SAVE OOF
    df = train.loc[valid_idx, ['customer_ID','target'] ].copy()
    df['oof_pred'] = oof_preds
    oof.append( df )
    
    
    if COMPUTE_PERM_IMPORTANCE:
            results = []
            print(' Computing Permutation feature importance...')
            
            # COMPUTE BASELINE (NO SHUFFLE)
            oof_preds = model.predict(dvalid)
            baseline_acc = amex_metric_mod(y_valid.values, oof_preds)
            results.append({'feature':'BASELINE','metric':baseline_acc})           

            for k in tqdm(range(len(FEATURES))):
                
                # SHUFFLE FEATURE K
                save_col = X_valid.iloc[:,k].copy()
                X_valid.iloc[:,k] = np.random.permutation(X_valid.iloc[:,k])
                
                dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
                        
                # COMPUTE OOF MAE WITH FEATURE K SHUFFLED
                oof_preds = model.predict(dvalid)
                acc = amex_metric_mod(y_valid.values, oof_preds)
                results.append({'feature':FEATURES[k],'metric':acc})
                X_valid.iloc[:,k] = save_col
         
            # DISPLAY XGB FEATURE IMPORTANCE
            print()
            df = pd.DataFrame(results)
            df = df.sort_values('metric', ascending = False)
            # SAVE XGB FEATURE IMPORTANCE
            df.to_csv(f'perm_feature_importance_fold_{fold+1}.csv',index=False)
            
            df = df.head(50)
            
            plt.figure(figsize=(10,20))
            plt.barh(np.arange(50),df.metric)
            plt.yticks(np.arange(50),df.feature.values)
            plt.title('XGB Permutation Feature Importance: Top 50',size=16)
            plt.xlim(0.7915, 0.7925)
            plt.ylim((-1,50))
            plt.plot([baseline_acc,baseline_acc],[-1,50], '--', color='orange',
                     label=f'Baseline OOF\nKaggle Metric={baseline_acc:.3f}')
            plt.xlabel(f'Fold {fold+1} OOF Kaggle Metric with feature permuted',size=14)
            plt.ylabel('Feature',size=14)
            plt.legend()
            plt.show()

    
    del dtrain, Xy_train, dd, df
    del X_valid, y_valid, dvalid, model
    _ = gc.collect()
    
    if ONE_FOLD_ONLY: break


# Interesting results, definitely looks there is room for improvement from the baseline by eliminating some features from the model 

# In[11]:


df = pd.read_csv(f'perm_feature_importance_fold_{fold+1}.csv')
print(df)

