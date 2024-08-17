#!/usr/bin/env python
# coding: utf-8

# # XGBoost Pyramid [CV 0.7968]
# 
# ## Pipeline:
# 1. Feature engineering: https://www.kaggle.com/code/roberthatch/amex-feature-engg-gpu-or-cpu-process-in-chunks
# 2. Training: https://www.kaggle.com/roberthatch/xgboost-pyramid-cv-0-7968
# 3. Test Predictions: (this notebook)
# 
# 
# Acknowledgements:
# 1. https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793
# 2. https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb 
# 
# 
# ## About the Pyramid:
# Main goal is to reduce over-specialization without the performance impact of DART on XGBoost GPU.
# 
# Start with a large forest with fewer number of rounds and higher learning rate to promote diversity that all the boosting will rely on.
# Between early layers, rescale the predictions to allow the next layer more room to impact. Again promoting more diversity. Rescale by less and less each layer.
# Slightly lower true learning rate per layer. True learning rate = learning_rate / num_parallel_tree.
# 
# Pyramid terminology and theory: Purely my own homebrew theory-crafting after reading about DART. If there's existing scholorly articles or prior work in this area, or if blending/stacking/whatever is a more correct term for what I'm doing, please let me know in the comments!

# # Load Libraries

# In[1]:


# ACTIVE = 'all'
# ACTIVE = 'train'
ACTIVE = 'test'


# In[2]:


# LOAD LIBRARIES
import pandas as pd, numpy as np # CPU libraries
import matplotlib.pyplot as plt, gc, os

GPU = True
try:
    import cupy, cudf
except ImportError:
    GPU = False

if GPU:
    print('RAPIDS version',cudf.__version__)
else:
    print("Disabling cudf, using pandas instead")
    cudf = pd


# In[3]:


# VERSION NAME FOR SAVED MODEL FILES
VER = 1
FEATURE_VER = 111

# RANDOM SEED
SEED = 108+5*VER+100*FEATURE_VER

# FOLDS PER MODEL
FOLDS = 5

# NOTEBOOK PATH
FEATURE_PATH = '../input/amex-feature-engg-gpu-or-cpu-process-in-chunks/'
MODEL_PATH = '../input/xgboost-pyramid-cv-0-7968/'

print("VER:", VER)
print("fVER:", FEATURE_VER)


# # Load Data
# Feature engineering is all done in: https://www.kaggle.com/code/roberthatch/amex-feature-engg-gpu-or-cpu-process-in-chunks

# In[4]:


if ACTIVE in ['train', 'all']:
    print('Reading train data...')
    TRAIN_PATH = f'{FEATURE_PATH}train_fe_v{FEATURE_VER}.parquet'
    train = pd.read_parquet(TRAIN_PATH)
    print(train.shape)

    train = train.sample(frac=1, random_state=SEED)
    train = train.reset_index(drop=True)
    train.head()


# # Train XGB
# We will train using `DeviceQuantileDMatrix`. This has a very small GPU memory footprint.

# In[5]:


# LOAD XGB LIBRARY
from sklearn.model_selection import KFold
import xgboost as xgb
print('XGB Version',xgb.__version__)


# XGB MODEL PARAMETERS
BASE_LEARNING_RATE = 0.01
xgb_params = { 
    'max_depth': 7,
    'subsample':0.75,
    'colsample_bytree': 0.35,
    'gamma':1.5,
    'lambda':70,
    'min_child_weight':8,

    'objective':'binary:logistic',
    'eval_metric':['logloss', 'auc'],  ## Early stopping is based on the last metric listed.
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'random_state':SEED,

    'num_parallel_tree':1
}


# In[6]:


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


# In[7]:


## TODO, replace with newer version. Not done yet to preserve exact tested score.

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

    print("  4%  :", top_four)
    print("  Gini:", gini[1]/gini[0])
    print("Kaggle:", 0.5 * (gini[1]/gini[0] + top_four))
    return 0.5 * (gini[1]/gini[0] + top_four)


# In[8]:


importances = []
PYRAMID_W = [0.5, 2/3, 0.75, 0.875, 1, 0]

def run_training(train, features):
    oof = []

    skf = KFold(n_splits=FOLDS)
    for fold,(train_idx, valid_idx) in enumerate(skf.split(
                train, train.target )):
        print('#'*25)
        print('### Fold',fold+1)
    
        # TRAIN, VALID, TEST FOR FOLD K
        X_valid = train.loc[valid_idx, features]
        y_valid = train.loc[valid_idx, 'target']

        print('### Train size',len(train_idx),'Valid size',len(valid_idx),'Valid positives',y_valid.sum())
        print(f'### Training with all of fold data...')
        print('#'*25)

        dvalid = xgb.DMatrix(data=X_valid, label=y_valid)

        # INFER XGB MODELS ON TEST DATA
        print(".")
        basic_score = 0
        for (layer, w) in enumerate(PYRAMID_W[:-1]):
            model = xgb.Booster()
            model.load_model(f'{MODEL_PATH}XGB_v{VER}_fold{fold}_layer{layer}.xgb')
            print(f'Loaded fold{fold}, layer{layer}')
            ptest = model.predict(dvalid, output_margin=True)

            ## reduce the impact of all model layers so far by w. This should be another way to reduce over-specialization, without the computational cost of DART
            if (w < 1.0):
                ptest = ptest * w

            ## This set_base_margin is what informs the next layer of the prior training.
            ## See code example from official demos: https://github.com/dmlc/xgboost/blob/master/demo/guide-python/boost_from_prediction.py
            dvalid.set_base_margin(ptest)

        layer = len(PYRAMID_W) - 1
        model = xgb.Booster()
        model.load_model(f'{MODEL_PATH}XGB_v{VER}_fold{fold}_layer{layer}.xgb')
        # INFER OOF FOLD K
        # Note: Not necessary with typical case ending pyramid with num_parallel_tree == 1, but more robust to divide best_ntree_limit by num_parallel_tree.
        #   Oddly, iteration range is based only on num_boost_rounds, but best_ntree_limit is stored as num_boost_rounds * num_parallel_trees
        print("Best_ntree_limit:", model.best_ntree_limit//xgb_params['num_parallel_tree'])
        oof_preds = model.predict(dvalid, iteration_range=(0,model.best_ntree_limit//xgb_params['num_parallel_tree']))
        print('For this fold:')
        ## TODO: update metric. Fork this notebook to confirm the latest version of the numpy implementation from author is even faster and equally accurate.
        ## https://www.kaggle.com/code/rohanrao/amex-competition-metric-implementations
        amex_metric_mod(y_valid.values, oof_preds)
    
        # SAVE OOF
        df = train.loc[valid_idx, ['customer_ID','target'] ].copy()
        df['oof_pred'] = oof_preds
        oof.append( df )
        
        del X_valid, y_valid, dvalid, model
        gc.collect()

    print('#'*25)
    print('OVERALL CV:')
    oof = pd.concat(oof,axis=0,ignore_index=True).set_index('customer_ID')
    amex_metric_mod(oof.target.values, oof.oof_pred.values)
    return oof


# In[9]:


if ACTIVE in ['train', 'all']:
    features = train.columns[1:-1]
    print(f'There are {len(features)} features!')
    print(train.shape)

    oof = run_training(train, features)

    # CLEAN RAM
    del train
    _ = gc.collect()

    oof_xgb = pd.read_parquet(TRAIN_PATH, columns=['customer_ID']).drop_duplicates()
    oof_xgb = oof_xgb.set_index('customer_ID')
    oof_xgb = oof_xgb.merge(oof, left_index=True, right_index=True)
    oof_xgb = oof_xgb.sort_index().reset_index(drop=True)
    oof_xgb.to_csv(f'oof_xgb_v{VER}.csv',index=False)
    oof_xgb.head()


# # Infer Test

# In[10]:


if ACTIVE in ['test', 'all']:
    gc.collect()

    # INFER TEST DATA IN PARTS

    TEST_SECTIONS = 2
    TEST_SUB_SECTIONS = 2

    test_preds = []
    customers = False
    for k in range(TEST_SECTIONS):
        for i in range(TEST_SUB_SECTIONS):    
            # READ PART OF TEST DATA
            print(f'\nReading test data...')
            test = cudf.read_parquet(f'{FEATURE_PATH}test{k}_fe_v{FEATURE_VER}.parquet')
            if i == 0:
                print(f'=> Test part {k+1} has shape', test.shape )
                if k == 0:
                    customers = test.index.copy()
                else:
                    customers = customers.append(test.index)

            # TEST DATA FOR XGB
            X_test = test
            n_rows = len(test.index)//TEST_SUB_SECTIONS
            print(".")
            if i+1 < TEST_SUB_SECTIONS:
                X_test = X_test.iloc[i*n_rows:(i+1)*n_rows, :].copy()
            elif TEST_SUB_SECTIONS > 1:
                X_test = X_test.iloc[i*n_rows:, :].copy()
            print(f'=> Test piece {k+1}, {i+1} has shape', X_test.shape )
            del test
            gc.collect()
            dtest = xgb.DMatrix(data=X_test)
            del X_test
            gc.collect()
            ## Need to reset to level 0 between folds.
            reset_margin = dtest.get_base_margin()

            # INFER XGB MODELS ON TEST DATA
            print(".")
            pred_folds = []
            for f in range(FOLDS):
                if (f > 0):
                    dtest.set_base_margin(reset_margin)
                for (layer, w) in enumerate(PYRAMID_W[:-1]):
                    model = xgb.Booster()
                    model.load_model(f'{MODEL_PATH}XGB_v{VER}_fold{f}_layer{layer}.xgb')
                    print(f'Loaded fold{f}, layer{layer}')
                    ptest = model.predict(dtest, output_margin=True)

                    ## reduce the impact of all model layers so far by w. This should be another way to reduce over-specialization, without the computational cost of DART
                    if (w < 1.0):
                        ptest = ptest * w

                    ## This set_base_margin is what informs the next layer of the prior training.
                    ## See code example from official demos: https://github.com/dmlc/xgboost/blob/master/demo/guide-python/boost_from_prediction.py
                    dtest.set_base_margin(ptest)

                layer = len(PYRAMID_W) - 1
                model = xgb.Booster()
                model.load_model(f'{MODEL_PATH}XGB_v{VER}_fold{f}_layer{layer}.xgb')
                print("Best_ntree_limit", model.best_ntree_limit//xgb_params['num_parallel_tree'])
                preds = model.predict(dtest, iteration_range=(0,model.best_ntree_limit//xgb_params['num_parallel_tree']))
                
                ## Create nested array to combine all predictions of a single fold together to rank them before averaging the predictions across folds.
                if f == len(test_preds):
                    test_preds.append([])
                test_preds[f].append(preds)

            # CLEAN MEMORY
            del dtest, model, reset_margin
            _ = gc.collect()


# In[11]:


def values_to_rank(p):
    u, v = np.unique(p, return_inverse=True)
    result = (np.cumsum(np.bincount(v)) - 1)[v]
    result = result.astype(np.float64) + p
    return result


def values_and_rank(p):
    u, v = np.unique(p, return_inverse=True)
    result = (np.cumsum(np.bincount(v)) - 1)[v]
    result = result / len(p)
    result = (result + p) / 2
    return result

if ACTIVE in ['test', 'all']:
    final_preds = []
    for preds in test_preds:
        preds = np.concatenate(preds)
        print(np.unique(preds).shape)
        preds = values_and_rank(preds)
        final_preds.append(preds)

    test_preds = final_preds[0]
    for i in range(1, len(final_preds)):
        test_preds += final_preds[i]
    print(np.unique(test_preds).shape[0])
    print(test_preds)
    print(test_preds.shape[0])


# # Create Submission CSV

# In[12]:


if ACTIVE in ['test', 'all']:
    # WRITE SUBMISSION FILE
    test = cudf.DataFrame(index=customers,data={'prediction':test_preds})
    sub = cudf.read_csv('../input/amex-default-prediction/sample_submission.csv')[['customer_ID']]
    if GPU:
        sub['customer_ID_hash'] = sub['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
        sub = sub.set_index('customer_ID_hash')
        sub = sub.merge(test[['prediction']], left_index=True, right_index=True, how='left')
        sub = sub.reset_index(drop=True)
    else:
        sub['customer_ID_hash'] = sub['customer_ID'].str[-16:].apply(int, base=16).astype('int64')
        sub = sub.set_index('customer_ID_hash')
        sub = pd.concat([sub, test], axis=1)
        sub = sub.dropna()
        sub = sub.reset_index(drop=True)

    # DISPLAY PREDICTIONS
    sub.to_csv(f'submission.csv',index=False)
    print('Submission file shape is', sub.shape )
    sub.head()

    # PLOT PREDICTIONS
    if GPU:
        sub = sub.to_pandas()
    plt.hist(sub.prediction, bins=100)
    plt.title('Test Predictions')
    plt.show()

