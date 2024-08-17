#!/usr/bin/env python
# coding: utf-8

# Part of the Feature Engineering is from the following notebooks:
# 
# [1-geomean-nn-and-6featlgbm-2-259-private-lb](https://www.kaggle.com/dkaraflos/1-geomean-nn-and-6featlgbm-2-259-private-lb) 
# 
# [physically-possible](https://www.kaggle.com/jazivxt/physically-possible)
# 
# [permutation-importance-for-feature-selection-part1](https://www.kaggle.com/corochann/permutation-importance-for-feature-selection-part1)
# 
# and the data is from:
# 
# [data-without-drift](https://www.kaggle.com/cdeotte/data-without-drift)

# In[1]:


from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from sklearn.metrics import f1_score, cohen_kappa_score, mean_squared_error
from logging import getLogger, Formatter, StreamHandler, FileHandler, INFO
from scipy.signal import butter, lfilter,filtfilt,savgol_filter
from sklearn.model_selection import KFold, GroupKFold
from tsfresh.feature_extraction import feature_calculators
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from pandas_profiling import ProfileReport
from tqdm import tqdm_notebook as tqdm
from contextlib import contextmanager
from joblib import Parallel, delayed
from IPython.display import display
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import signal
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import random as rn
import pandas as pd
import numpy as np
import scipy as sp
import itertools
import warnings
import librosa
import time
import pywt
import os
import gc


warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


BATCHSIZE = 50000
SEED = 529
SELECT = True
SPLITS = 5
fe_config =[
    (True, [5000, 10000, 50000], True, True, False),
    ]


# In[3]:


def init_logger():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter(LOGFORMAT))
    fh_handler = FileHandler('{}.log'.format(MODELNAME))
    fh_handler.setFormatter(Formatter(LOGFORMAT))
    logger.setLevel(INFO)
    logger.addHandler(handler)
    logger.addHandler(fh_handler)
    


# In[4]:


@contextmanager
def timer(name : Text):
    t0 = time.time()
    yield
    logger.info(f'[{name}] done in {time.time() - t0:.0f} s')

COMPETITION = 'ION-Switching'
logger = getLogger(COMPETITION)
LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'
MODELNAME = 'Baseline'


# In[5]:


def seed_everything(seed : int) -> NoReturn :
    
    rn.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
seed_everything(SEED)


# In[6]:


def read_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train = pd.read_csv('../input/data-without-drift/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int8})
    test  = pd.read_csv('../input/data-without-drift/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
    sub  = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})
    return train, test, sub



# In[7]:


tr_visual_idx = [0, 500000, 600000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000]
te_visual_idx = [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1500000, 2000000]
tr_data_type = [0,0,0,1,2,4,3,1,2,3,4]
te_data_type = [0,2,3,0,1,4,3,4,0,2,0,0]

def add_visual_batching(df: pd.DataFrame, train: bool = True):
    idx = tr_visual_idx if train else te_visual_idx
    type_ = tr_data_type if train else te_data_type
    s = idx; t = type_
    visual_batch = np.zeros((s[-1],), dtype=np.int64)
    visual_type = np.zeros((s[-1],), dtype=np.int64)
    df['visual_batch'] = 0
    df['visual_type'] = 0
    
    for i, j in zip(range(len(s) - 1), t):
        df.loc[s[i]:s[i+1],'visual_batch'] = i
        df.loc[s[i]:s[i+1],'visual_type'] = j
    return df


# In[8]:


def reduce_mem_usage(df: pd.DataFrame,
                     verbose: bool = True) -> pd.DataFrame:
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if (c_min > np.iinfo(np.int8).min
                        and c_max < np.iinfo(np.int8).max):
                    df[col] = df[col].astype(np.int8)
                elif (c_min > np.iinfo(np.int16).min
                      and c_max < np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)
                elif (c_min > np.iinfo(np.int32).min
                      and c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min
                      and c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min
                      and c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem

    msg = f'Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)'
    if verbose:
        print(msg)

    return df


# In[9]:


def filter_wave(x, cutoff=(-1, -1), N=4, filtering='lfilter'):

    assert x.ndim == 1
    output = 'sos' if filtering == 'sos' else 'ba'
    if cutoff[0] <= 0 and cutoff[1] <= 0:
        # Do not apply filter
        return x
    elif cutoff[0] <= 0 and cutoff[1] > 0:
        # Apply low pass filter
        output = signal.butter(N, Wn=cutoff[1]/len(x), btype='lowpass', output=output)
    elif cutoff[0] > 0 and cutoff[1] <= 0:
        # Apply high pass filter
        output = signal.butter(N, Wn=cutoff[0]/len(x), btype='highpass', output=output)
    else:
        # Apply band pass filter
        output = signal.butter(N, Wn=(cutoff[0]/len(x), cutoff[1]/len(x)), btype='bandpass', output=output)

    if filtering == 'lfilter':
        b, a = output
        return signal.lfilter(b, a, x)
    elif filtering == 'filtfilt':
        b, a = output
        return signal.filtfilt(b, a, x)
    elif filtering == 'sos':
        sos = output
        return signal.sosfilt(sos, x)
    else:
        raise ValueError("[ERROR] Unexpected value filtering={}".format(filtering))


# In[10]:


def lag_with_pct_change(df : pd.DataFrame,
                        shift_sizes : Optional[List]=[1, 2, 3],
                        add_pct_change : Optional[bool]=False,
                        add_pct_change_lag : Optional[bool]=False) -> pd.DataFrame:
    df['batch'] = df.index // 500000
    df['batch_idx'] = df.index - (df.batch * 500000)
    smooth_sig = lambda x: filter_wave(x, cutoff=(0, 100), filtering='filtfilt')
    df['baseline'] = np.concatenate(df.groupby(['visual_batch'])['signal'].apply(smooth_sig))
        
    for shift_size in shift_sizes:    
        df['signal_shift_pos_'+str(shift_size)] = df.groupby(['batch']).shift(shift_size, fill_value=0.)['signal']
        df['signal_shift_neg_'+str(shift_size)] = df.groupby(['batch']).shift(-1*shift_size, fill_value=0.)['signal']

    if add_pct_change:
        df['pct_change'] = df['signal'].pct_change()
        if add_pct_change_lag:
            for shift_size in shift_sizes:    
                df['pct_change_shift_pos_'+str(shift_size)] = df.groupby(['batch']).shift(shift_size, fill_value=0.)['pct_change']
                df['pct_change_shift_neg_'+str(shift_size)] = df.groupby(['batch']).shift(-1*shift_size, fill_value=0.)['pct_change']
    return df


# In[11]:


def feature_enginering_by_batch(z : Union[pd.Series, np.array],
                                window_size : Optional[List]=[50000]) -> pd.DataFrame:
    
    temp = pd.DataFrame(index=[0], dtype=np.float16)
    if window_size is not None:
        for window in window_size:
            roll = pd.Series(z).rolling(window=window, min_periods=1, center=True)
            temp[f'roll_mean_{window}'] = roll.mean()
            temp[f'roll_max_{window}'] = roll.max()
            temp[f'roll_min_{window}'] = roll.min()
            temp[f'roll_std_{window}'] = roll.std()
            temp[f'roll_mean_abs_chg_{window}'] = roll.apply(lambda x: np.mean(np.abs(np.diff(x))), raw=True)
            temp[f'roll_abs_max_{window}'] = roll.apply(lambda x: np.max(np.abs(x)), raw=True)
            temp[f'roll_abs_min_{window}'] = roll.apply(lambda x: np.min(np.abs(x)), raw=True)
            temp[f'roll_range_{window}'] = temp[f'roll_max_{window}'] - temp[f'roll_min_{window}']
            temp[f'roll_max_to_min_{window}'] = temp[f'roll_max_{window}'] / temp[f'roll_min_{window}']
            temp[f'roll_abs_avg_{window}'] = (temp[f'roll_abs_max_{window}'] + temp[f'roll_abs_min_{window}']) / 2
    
    for i in range(4, 5): 
        temp[f'kstat_{i}'] = stats.kstat(z, i)

    for i in range(4, 5):
        temp[f'moment_{i}'] = stats.moment(z, i)    
           
    return temp


# In[12]:


def parse_sample(sample : pd.DataFrame,
                 batch_no : int,
                 window_size : List) -> pd.DataFrame:
    
    temp = feature_enginering_by_batch(sample['signal'].values, window_size)
    temp['visual_batch'] = int(batch_no)
    
    return temp


# In[13]:


def sample_gen(df : pd.DataFrame,
               window_size : List,
               batches : List=[0], ) -> pd.DataFrame:
    
    result = Parallel(n_jobs=1, temp_folder='/tmp', max_nbytes=None, backend='multiprocessing')(delayed(parse_sample)
                                              (df[df['visual_batch']==i], int(i), window_size)
                                                                                              for i in tqdm(batches))
    data = [r.values for r in result]
    data = np.vstack(data)
    cols = result[0].columns
    X = pd.DataFrame(data, columns=cols)
    X = reduce_mem_usage(X, False)
    X = X.sort_values('visual_batch')
    
    return X


# In[14]:


def run_feat_enginnering(df : pd.DataFrame,
                         create_all_data_feats : bool,
                         window_size : List,
                         add_visual_batch : bool,
                         is_train: Optional[bool]=None) -> pd.DataFrame:
    
    if add_visual_batching:
        df = add_visual_batching(df, is_train)
    if create_all_data_feats:
        df = lag_with_pct_change(df, [1, 2, 3],  add_pct_change=True, add_pct_change_lag=True)
    batches = df['visual_batch'].unique().tolist()
    batch_feats=sample_gen(df, window_size=window_size, batches=batches)
    df = pd.merge(df, batch_feats, on='visual_batch', how='left')
    df = reduce_mem_usage(df, False)
    
    return df


# In[15]:


def feature_selection(df : pd.DataFrame,
                      df_test : pd.DataFrame,
                      subtract_only : Optional[bool]=True,
                      idx_cols : List=['time'],
                      target_col : List=['open_channels']) -> Tuple[pd.DataFrame , pd.DataFrame]:
    
    drops = df.columns[df.isna().sum()>25000]
    df = df.drop(drops, axis=1)
    df_test = df_test.drop(drops, axis=1)
  
    gc.collect()
    if subtract_only == False:
        corrcoef_cols = [col for col in df.columns.tolist() if col not in (idx_cols+target_col)]
        first=dict(); second=dict(); third=dict()
        for col in corrcoef_cols:
            ss = np.corrcoef(df[col], df['open_channels'])[0, 1]
            first[col] = ss
            ss = np.corrcoef(df[col]-df['signal'], df['open_channels'])[0, 1]
            second[col] = ss
            ss = np.corrcoef(df[col]*df['signal'], df['open_channels'])[0, 1]
            third[col] = ss
        corr_df = pd.DataFrame.from_dict(
            {
            'Base':first, 
            'Signal-Subtracted': second,
            'Signal-Multiplied': third
            }
        ).fillna(0).apply(np.abs).sort_values('Base', ascending=False)

        base_cols = corr_df.sort_values('Base', ascending=False).head(100).index.tolist()
        multiply_cols = corr_df.sort_values('Signal-Multiplied', ascending=False).head(10).index.tolist()
        subtract_cols = corr_df.sort_values('Signal-Subtracted', ascending=False).head(25).index.tolist()
        display(corr_df.sort_values('Base', ascending=False).tail(50))
        all_cols = list(set(base_cols + multiply_cols + subtract_cols + idx_cols + target_col))
        all_cols_test = list(set(base_cols + multiply_cols + subtract_cols + idx_cols))   
        drops = list(set(multiply_cols + subtract_cols)-set(base_cols))
        df = df[all_cols]
        df_test = df_test[all_cols_test]
    
        for col in multiply_cols:
            df[col+'_m'] = df[col] * df['signal']
            df_test[col+'_m'] = df_test[col] * df_test['signal']        
        for col in subtract_cols:
            df[col+'_s'] = df[col] - df['signal']
            df_test[col+'_s'] = df_test[col] - df_test['signal']
        df = df.drop(drops, axis=1)
    else:
        not_imp = ['kstat_4', 'moment_4','signal','baseline']
        subtract_cols = list(set(df.columns.tolist())-set(idx_cols + target_col + not_imp))
        for col in subtract_cols:
            df[col+'_s'] = df[col] - df['signal']
            df_test[col+'_s'] = df_test[col] - df_test['signal']
            df[col+'_b'] = df[col] - df['baseline']
            df_test[col+'_b'] = df_test[col] - df_test['baseline']
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(-1)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.fillna(-1) 
    df = reduce_mem_usage(df, False)
    df_test = reduce_mem_usage(df_test, False)
    gc.collect()
    return df, df_test


# In[16]:


def MacroF1Metric(preds : np.array, dtrain : lgb.Dataset) -> Tuple[Text, np.array, bool] :
    
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(int)
    score = f1_score(labels, preds, average = 'macro')
    
    return ('MacroF1Metric', score, True)


# In[17]:


def get_params(seed : int) -> Dict :
    params = dict()
    params['learning_rate']=0.1;
    params['max_depth']=-1;
    params['num_leaves']=128;
    params['metric']='l1';
    params['random_state']=seed;
    params['n_jobs']=-1;
    params['feature_fraction']=1 ;
    params['boosting']='gbdt';
    params['bagging_seed']=seed;
    params['bagging_freq']=5;
    params['bagging_fraction']=0.8;
    params['reg_alpha']=0.1;
    params['reg_lambda']=0.3
    return params


# In[18]:


def run_cv_model_by_batch(train : pd.DataFrame,
                          test : pd.DataFrame,
                          splits : int,
                          shuffle : bool,
                          seed : int,
                          params : Dict,
                          feats : List,
                          sample_submission: pd.DataFrame) -> pd.DataFrame:
    
    oof_ = np.zeros(len(train))
    preds_ = np.zeros(len(test))
    target = ['open_channels']
    imp_df = pd.DataFrame(index=feats)
    groups = np.tile(np.arange(splits).repeat(500000 // splits), 10)
    kf = GroupKFold(n_splits=splits)
    for n_fold, (tr_idx, val_idx) in enumerate(kf.split(train, train[target], groups=groups)):
        trn_data = lgb.Dataset(train[feats].iloc[tr_idx], label=train[target].iloc[tr_idx])
        val_data = lgb.Dataset(train[feats].iloc[val_idx], label=train[target].iloc[val_idx])
        
        model = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=0 , early_stopping_rounds = 50)
        oof_[val_idx] += model.predict(train[feats].iloc[val_idx], num_iteration=model.best_iteration)
        preds_ += model.predict(test[feats], num_iteration=model.best_iteration) / SPLITS
        f1_score_ = f1_score(train[target].iloc[val_idx], np.round(np.clip(oof_[val_idx], 0, 10)).astype(int), average = 'macro')
        rmse_score_ = np.sqrt(mean_squared_error(train[target].iloc[val_idx], oof_[val_idx]))
        logger.info(f'Fold {n_fold + 1} macro f1 score : {f1_score_ :1.5f} rmse score : {rmse_score_:1.5f}')
        imp_df[f'feat_importance_{n_fold + 1}'] = model.feature_importance(importance_type='gain')
    f1_score_ = f1_score(train[target], np.round(np.clip(oof_, 0, 10)).astype(int), average = 'macro')
    rmse_score_ = np.sqrt(mean_squared_error(train[target], oof_))
    logger.info(f'OOF macro f1 score : {f1_score_:1.5f} oof rmse score : {rmse_score_:1.5f}')
    sample_submission['open_channels'] = np.round(np.clip(preds_, 0, 10)).astype(int)
    sample_submission.to_csv('submission.csv', index=False, float_format='%.4f')
    display(sample_submission.head())
    np.save('oof.npy', oof_)
    np.save('preds.npy', preds_)

    return imp_df


# In[19]:


def run_everything(fe_config : List) -> NoReturn:
    idx_cols = ['time','batch','batch_idx','visual_batch']
    target_col = ['open_channels']
    type_col = ['visual_type']
    init_logger()
    with timer(f'Reading Data'):
        logger.info('Reading Data Started ...')
        train, test, sample_submission = read_data()
        logger.info('Reading and Cleaning Data Completed ...')
        
    with timer(f'Creating Features'):
        logger.info('Feature Enginnering Started ...')
        for config in fe_config:
            train = run_feat_enginnering(train, create_all_data_feats=config[0], window_size=config[1], add_visual_batch=config[2], is_train=config[3])
            test  = run_feat_enginnering(test,  create_all_data_feats=config[0], window_size=config[1], add_visual_batch=config[2], is_train=config[4])
        if SELECT:
            train, test = feature_selection(train, test, subtract_only=True, idx_cols=idx_cols+type_col, target_col=target_col)
        logger.info('Feature Enginnering Completed ...')

    with timer(f'Running LGB model'):
        logger.info(f'Training LGB model with {SPLITS} folds Started ...')
        params = get_params(SEED)
        feats = [c for c in train.columns if c not in (idx_cols+target_col)]
        imp = run_cv_model_by_batch(train, test, splits=SPLITS, shuffle=True, seed=SEED ,params=params, feats=feats, sample_submission=sample_submission)
        logger.info(f'Training completed ...')
    
    return imp


# In[20]:


imp = run_everything(fe_config)


# In[21]:


imp

