#!/usr/bin/env python
# coding: utf-8

# This is a direct copy of the followign kernel, with LGBM replaced with HistGradientBoosting
# 
# https://www.kaggle.com/siavrez/fe-pipeline

# Part of the Feature Engineering is from the following notebooks:
# 
# [1-geomean-nn-and-6featlgbm-2-259-private-lb](https://www.kaggle.com/dkaraflos/1-geomean-nn-and-6featlgbm-2-259-private-lb) 
# 
# [physically-possible](https://www.kaggle.com/jazivxt/physically-possible)

# In[1]:


from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict

from sklearn.metrics import f1_score, cohen_kappa_score, mean_squared_error
from logging import getLogger, Formatter, StreamHandler, FileHandler, INFO
from scipy.signal import butter, lfilter,filtfilt,savgol_filter
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
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
fe_config = [
    (True, True, 50000, None),
    (False, False, 5000, None),
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


def read_data(base : os.path.abspath) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    train = pd.read_csv(os.path.join(base + '/train.csv'), dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int8})
    test  = pd.read_csv(os.path.join(base + '/test.csv'), dtype={'time': np.float32, 'signal': np.float32})
    sub  = pd.read_csv(os.path.join(base + '/sample_submission.csv'), dtype={'time': np.float32})
    
    return train, test, sub




# In[7]:


def batching(df : pd.DataFrame,
             batch_size : int,
             add_index : Optional[bool]=True) -> pd.DataFrame :
    
    df['batch_'+ str(batch_size)] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values + 1
    df['batch_'+ str(batch_size)] = df['batch_'+ str(batch_size)].astype(np.uint16)
    if add_index:
        df['batch_' + str(batch_size) +'_idx'] = df.index  - (df['batch_'+ str(batch_size)] * batch_size)
        df['batch_' + str(batch_size) +'_idx'] = df['batch_' + str(batch_size) +'_idx'].astype(np.uint16)
        
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


def maddest(d : Union[np.array, pd.Series, List], axis : Optional[int]=None) -> np.array:  
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


# In[10]:


def denoise_signal(x : Union[np.array, pd.Series],
                   wavelet : Optional[Text]='db4',
                   level : Optional[int]=1) -> np.array:
    
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode='per')


# In[11]:


def denoise_signal_simple(x : Union[np.array, pd.Series],
                          wavelet : Optional[Text]='db4',
                          level : Optional[int]=1) -> np.array:
    
    coeff = pywt.wavedec(x, wavelet, mode="per")
    uthresh = 10
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    
    return pywt.waverec(coeff, wavelet, mode='per')


# In[12]:


def trend(df : Union[pd.Series, np.array],
          abs_values: Optional[bool]=False) -> float:
    
    idx = np.array(range(len(df)))
    if abs_values:
        df = np.abs(df)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), df)
    
    return lr.coef_[0]


# In[13]:


def change_rate(df : Union[pd.Series, np.array]) -> float:
    
    change = (np.diff(df) / df[:-1])
    change = change[np.nonzero(change)[0]]
    change = change[~np.isnan(change)]
    change = change[change != -np.inf]
    change = change[change != np.inf]
    
    return np.mean(change)


# In[14]:


def lag_with_pct_change(df : pd.DataFrame,
                        batch_size : int,
                        shift_sizes : Optional[List]=[1, 2],
                        add_pct_change : Optional[bool]=False,
                        add_pct_change_lag : Optional[bool]=False) -> pd.DataFrame:
    
    assert 'batch_' + str(batch_size) +'_idx' in df.columns
    for shift_size in shift_sizes:    
        df['signal_shift_pos_'+str(shift_size)] = df['signal'].shift(shift_size).fillna(0)
        df['signal_shift_neg_'+str(shift_size)] = df['signal'].shift(-1*shift_size).fillna(0)
        for i in df[df['batch_' + str(batch_size) +'_idx'].isin(range(shift_size))].index:
            df['signal_shift_pos_'+str(shift_size)][i] = np.nan
        for i in df[df['batch_' + str(batch_size) +'_idx'].isin(range(batch_size - shift_size, batch_size))].index:
            df['signal_shift_neg_'+str(shift_size)][i] = np.nan
    if add_pct_change:
        df['pct_change'] = df['signal'].pct_change()
        if add_pct_change_lag:
            df['pct_change_shift_pos_'+str(shift_size)] = df['pct_change'].shift(shift_size).fillna(0)
            df['pct_change_shift_neg_'+str(shift_size)] = df['pct_change'].shift(-1*shift_size).fillna(0)
            for i in df[df['batch_' + str(batch_size) +'_idx'].isin(range(shift_size))].index:
                df['pct_change_shift_pos_'+str(shift_size)][i] = np.nan
            for i in df[df['batch_' + str(batch_size) +'_idx'].isin(range(batch_size - shift_size, batch_size))].index:
                df['pct_change_shift_neg_'+str(shift_size)][i] = np.nan 
    return df


# In[15]:


def feature_enginering_by_batch(z : Union[pd.Series, np.array],
                                batch_size : int,
                                window_size : Optional[List]=None) -> pd.DataFrame:
    
    temp = pd.DataFrame(index=[0], dtype=np.float16)
    
    temp['mean'] = z.mean()
    temp['max'] = z.max()
    temp['min'] = z.min()
    temp['std'] = z.std()  
    temp['mean_abs_chg'] = np.mean(np.abs(np.diff(z)))
    temp['abs_max'] = np.max(np.abs(z))
    temp['abs_min'] = np.min(np.abs(z))
    temp['range'] = temp['max'] - temp['min']
    temp['max_to_min'] = temp['max'] / temp['min']
    temp['abs_avg'] = (temp['abs_max'] + temp['abs_min']) / 2
    
    for i in range(1, 5): 
        temp[f'kstat_{i}'] = stats.kstat(z, i)

    for i in range(2, 5):
        temp[f'moment_{i}'] = stats.moment(z, i)

    for i in [1, 2]:
        temp[f'kstatvar_{i}'] = stats.kstatvar(z, i)
    
    if window_size is not None:
        for window in window_size:
            temp['percentile_roll_'+str(window)+'_std_25'] = np.percentile(pd.Series(z).rolling(window).std().dropna().values, 25)
            temp['percentile_roll_'+str(window)+'_std_75'] = np.percentile(pd.Series(z).rolling(window).std().dropna().values, 75)
            temp['percentile_roll_'+str(window)+'_std_05'] = np.percentile(pd.Series(z).rolling(window).std().dropna().values,  5)
            temp['percentile_roll_'+str(window)+'_std_95'] = np.percentile(pd.Series(z).rolling(window).std().dropna().values, 95)
            temp['percentile_roll_'+str(window)+'_mean_25'] = np.percentile(pd.Series(z).rolling(window).mean().dropna().values, 25)
            temp['percentile_roll_'+str(window)+'_mean_75'] = np.percentile(pd.Series(z).rolling(window).mean().dropna().values, 75)
            temp['percentile_roll_'+str(window)+'_mean_05'] = np.percentile(pd.Series(z).rolling(window).mean().dropna().values,  5)
            temp['percentile_roll_'+str(window)+'_mean_95'] = np.percentile(pd.Series(z).rolling(window).mean().dropna().values, 95)            
    return temp


# In[16]:


def parse_sample(sample : pd.DataFrame,
                 batch_no : int,
                 batch_size : int,
                 window_size : List) -> pd.DataFrame:
    
    temp = feature_enginering_by_batch(sample['signal'].values, batch_size, window_size)
    temp['batch_'+ str(batch_size)] = int(batch_no)
    
    return temp


# In[17]:


def sample_gen(df : pd.DataFrame,
               batch_size : int,
               window_size : List,
               batches : List=[0], ) -> pd.DataFrame:
    
    result = Parallel(n_jobs=1, temp_folder='/tmp', max_nbytes=None, backend='multiprocessing')(delayed(parse_sample)
                                              (df[df['batch_'+ str(batch_size)]==i], int(i), batch_size, window_size)
                                                                                              for i in tqdm(batches))
    data = [r.values for r in result]
    data = np.vstack(data)
    cols = result[0].columns
    cols = [name+'_'+str(batch_size) if name!='batch_'+ str(batch_size) else 'batch_'+ str(batch_size) for name in cols ]
    X = pd.DataFrame(data, columns=cols)
    X = reduce_mem_usage(X, False)
    X = X.sort_values('batch_'+ str(batch_size))
    
    return X


# In[18]:


def run_feat_enginnering(df : pd.DataFrame,
                         create_all_data_feats : bool,
                         add_index : bool,
                         batch_size : int,
                         window_size : List) -> pd.DataFrame:
    
    df = batching(df, batch_size=batch_size, add_index=add_index)
    if create_all_data_feats:
        df = lag_with_pct_change(df, batch_size, [1, 2, 4],  add_pct_change=True, add_pct_change_lag=True)
    batches = df['batch_'+ str(batch_size)].unique().tolist()
    batch_feats=sample_gen(df, batch_size=batch_size, window_size=window_size, batches=batches)
    df = pd.merge(df, batch_feats, on='batch_'+ str(batch_size), how='left')
    df = reduce_mem_usage(df, False)
    
    return df


# In[19]:


def feature_selection(df : pd.DataFrame,
                      df_test : pd.DataFrame,
                      subtract_only : Optional[bool]=True,
                      idx_cols : List=['time'],
                      target_col : List=['open_channels']) -> Tuple[pd.DataFrame , pd.DataFrame]:
    
    drops = df.columns[df.isna().sum()>25000]
    df = df.drop(drops, axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
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
        not_imp = ['kstat_1_5000', 'kstat_2_5000', 'kstat_3_5000', 'kstat_4_5000', 'moment_2_5000',
                   'moment_3_5000','moment_4_5000', 'kstatvar_1_5000', 'kstatvar_2_5000','kstat_1_50000',
                   'kstat_2_50000', 'kstat_3_50000', 'kstat_4_50000', 'moment_2_50000', 'moment_3_50000',
                   'moment_4_50000', 'kstatvar_1_50000', 'kstatvar_2_50000']
        subtract_cols = list(set(df.columns.tolist())-set(idx_cols + target_col + not_imp))
        for col in subtract_cols:
            df[col+'_s'] = df[col] - df['signal']
            df_test[col+'_s'] = df_test[col] - df_test['signal']
    df = reduce_mem_usage(df, False)
    df_test = reduce_mem_usage(df_test, False)

    gc.collect()
    return df, df_test


# In[20]:


def MacroF1Metric(preds : np.array, dtrain : lgb.Dataset) -> Tuple[Text, np.array, bool] :
    
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(int)
    score = f1_score(labels, preds, average = 'macro')
    
    return ('MacroF1Metric', score, True)


# In[21]:


def run_cv_model_by_batch(train : pd.DataFrame,
                          test : pd.DataFrame,
                          splits : int,
                          shuffle : bool,
                          seed : int,
                          batch_col : Text,
                          params : Dict,
                          feats : List,
                          sample_submission: pd.DataFrame) -> pd.DataFrame:
    
    oof_ = np.zeros(len(train))
    preds_ = np.zeros(len(test))
    target = ['open_channels']
    imp_df = pd.DataFrame(index=feats)
    kf = KFold(splits, shuffle, seed)
    for n_fold, (tr_idx, val_idx) in enumerate(kf.split(train, train[target], groups=train[batch_col])):
        tr_x = train[feats].iloc[tr_idx]
        vl_x = train[feats].iloc[val_idx]
        tr_y = train[target].iloc[tr_idx].values
        vl_y = train[target].iloc[val_idx].values
        model = HistGradientBoostingRegressor(learning_rate = 0.1, max_iter=800, random_state = 404, validation_fraction=None, verbose = 0, max_depth=12, min_samples_leaf=25, l2_regularization=0.05)
        model.fit(tr_x, tr_y)
        oof_[val_idx] += model.predict(train[feats].iloc[val_idx])
        preds_ += model.predict(test[feats]) / SPLITS
        f1_score_ = f1_score(train[target].iloc[val_idx], np.round(np.clip(oof_[val_idx], 0, 10)).astype(int), average = 'macro')
        rmse_score_ = np.sqrt(mean_squared_error(train[target].iloc[val_idx], oof_[val_idx]))
        logger.info(f'Training fold {n_fold + 1} completed. macro f1 score : {f1_score_ :1.5f} rmse score : {rmse_score_:1.5f}')
        #imp_df[f'feat_importance_{n_fold + 1}'] = model.feature_importance(importance_type='gain')
    f1_score_ = f1_score(train[target], np.round(np.clip(oof_, 0, 10)).astype(int), average = 'macro')
    rmse_score_ = np.sqrt(mean_squared_error(train[target], oof_))
    logger.info(f'Training completed. oof macro f1 score : {f1_score_:1.5f} oof rmse score : {rmse_score_:1.5f}')
    sample_submission['open_channels'] = np.round(np.clip(preds_, 0, 10)).astype(int)
    sample_submission.to_csv('submission.csv', index=False, float_format='%.4f')
    display(sample_submission.head())
    np.save('oof.npy', oof_)
    np.save('preds.npy', preds_)

    return imp_df


# In[22]:


def get_params(seed : int) -> Dict :
    params = dict()
    params['learning_rate']=0.009;
    params['max_depth']=-1;
    params['num_leaves']=257;
    params['metric']='rmse';
    params['random_state']=seed;
    params['n_jobs']=-1;
    params['feature_fraction']=1 ;
    params['boosting']='goss';
    params['boost_from_average']=True;
    params['bagging_seed']=seed;
    params['bagging_freq']=0;
    params['bagging_fraction']=1;
    params['reg_alpha']=0;
    params['reg_lambda']=0
    params['force_row_wise']=True
    return params


# In[23]:


def run_everything(fe_config : List) -> NoReturn:
    not_feats_cols = ['time']
    target_col = ['open_channels']
    init_logger()
    with timer(f'Reading Data'):
        logger.info('Reading Data Started ...')
        base = os.path.abspath('/kaggle/input/liverpool-ion-switching/')
        train, test, sample_submission = read_data(base)
        logger.info('Reading Data Completed ...')
        
    with timer(f'Creating Features'):
        logger.info('Feature Enginnering Started ...')
        for config in fe_config:
            train = run_feat_enginnering(train, create_all_data_feats=config[0], add_index=config[1], batch_size=config[2], window_size=config[3])
            test  = run_feat_enginnering(test,  create_all_data_feats=config[0], add_index=config[1], batch_size=config[2], window_size=config[3])
            not_feats_cols.append('batch_'+str(config[2]))
            if config[1]:
                not_feats_cols.append('batch_'+str(config[2])+'_idx')
        if SELECT:
            train, test = feature_selection(train, test, subtract_only=True, idx_cols=not_feats_cols, target_col=target_col)
        logger.info('Feature Enginnering Completed ...')

    with timer(f'Running HistGradientBoosting model'):
        logger.info(f'Training HistGradientBoosting model with {SPLITS} folds Started ...')
        params = get_params(SEED)
        feats = [c for c in train.columns if c not in (not_feats_cols+target_col)]
        importances = run_cv_model_by_batch(train, test, splits=SPLITS, shuffle=True, seed=SEED, batch_col='batch_50000',params=params, feats=feats, sample_submission=sample_submission)
        importances.to_csv('importances.csv')
        logger.info(f'Training completed ...')


# In[24]:


run_everything(fe_config)


# In[ ]:




