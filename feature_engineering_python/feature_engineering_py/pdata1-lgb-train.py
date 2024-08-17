#!/usr/bin/env python
# coding: utf-8

# 1. pdata1-features1:feature engineering https://www.kaggle.com/quincyqiang/pdata1-features1
# 2. pdata1-lgb-train：current notebook  https://www.kaggle.com/quincyqiang/pdata1-lgb-train
# 2. pdata1-lgb-inference：inference https://www.kaggle.com/quincyqiang/pdata1-features1

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm 
from glob import glob
from imp import reload
import copy as cp
import sys
sys.path.append('../input/features-util')
# from utils import util
import util
reload(util)
import joblib


# In[2]:


import copy as cp
from glob import glob
from imp import reload
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from sklearn import model_selection
import os


def RMSPEMetric(XGBoost=False):
    def RMSPE(yhat, dtrain, XGBoost=XGBoost):

        y = dtrain.get_label()
        elements = ((y - yhat) / y) ** 2
        if XGBoost:
            return 'RMSPE', float(np.sqrt(np.sum(elements) / len(y)))
        else:
            return 'RMSPE', float(np.sqrt(np.sum(elements) / len(y))), False

    return RMSPE


reload(util)


if __name__ == '__main__':

    path_lst = glob('../input/optiver-realized-volatility-prediction/book_train.parquet/*')
    stock_lst = [os.path.basename(path).split('=')[-1] for path in path_lst]

    print(len(stock_lst))
    data_type = 'train'
    fe_df = pd.read_pickle('../input/pdata1-features1/train_stock_df.pkl')
    train = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv')

    fe_df = fe_df.merge(
        train
        , how='left'
        , on=['stock_id', 'time_id']
    ).replace([np.inf, -np.inf], np.nan).fillna(method='ffill')

    # for name in tqdm(name_lst):
    #     ret = util.gen_data(name)
    #     ret.to_csv('../data/20210731/{}.csv'.format())

    # LightGBM parameters
    params = {
        'n_estimators': 10000,
        'objective': 'rmse',
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'learning_rate': 0.01,
        'subsample': 0.72,
        'subsample_freq': 4,
        'feature_fraction': 0.8,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'seed': 46,
        'early_stopping_rounds': 300,
        'verbose': -1
    }

    data = fe_df
    label = fe_df['target']
    features = fe_df.columns.difference(['time_id', 'target']).tolist()
    data_ = fe_df[features]
    cats = ['stock_id', ]
    X_train = data_.reset_index(drop=True)
    y_train = label
    # y_train = pd.DataFrame(label_)
    models = []
    oof_df = fe_df[['time_id', 'stock_id']].copy()
    oof_df['target'] = y_train
    oof_df['pred'] = np.nan

    cv = model_selection.KFold(n_splits=10,
                               shuffle=True,
                               random_state=666)

    kf = cv.split(X_train, y_train)

    fi_df = pd.DataFrame()
    fi_df['features'] = features
    fi_df['importance'] = 0

    for fold_id, (train_index, valid_index) in tqdm(enumerate(kf)):
        # split
        X_tr = X_train.loc[train_index, features]
        X_val = X_train.loc[valid_index, features]
        y_tr = y_train.loc[train_index].values.reshape(-1)
        y_val = y_train.loc[valid_index].values.reshape(-1)

        # model (note inverse weighting)
        train_set = lgb.Dataset(X_tr,
                                y_tr,
                                categorical_feature=cats,
                                weight=1 / np.power(y_tr, 2))
        val_set = lgb.Dataset(X_val,
                              y_val,
                              categorical_feature=cats,
                              weight=1 / np.power(y_val, 2))
        model = lgb.train(params,
                          train_set,
                          valid_sets=[train_set, val_set],
                          feval=RMSPEMetric(),
                          verbose_eval=250)

        # feature importance
        fi_df[f'importance_fold{fold_id}'] = model.feature_importance(
            importance_type="gain")
        fi_df['importance'] += fi_df[f'importance_fold{fold_id}'].values

        # save model
        joblib.dump(model, f'model_fold{fold_id}.pkl')
        print('model saved!')



