#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datatable as dt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from category_encoders import *
import gc


# In[2]:


from warnings import filterwarnings
filterwarnings('ignore')


# # Memory Reduction
# 
# * This memory reduction part taken from https://www.kaggle.com/azzamradman/tps-10-single-xgboost/notebook
#   amazing notebook. Please upvote it if you like this part.

# In[3]:


def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


# In[4]:


get_ipython().run_cell_magic('time', '', "train = dt.fread('../input/tabular-playground-series-oct-2021/train.csv').to_pandas().drop('id', axis=1)\ntrain = reduce_memory_usage(train)\ntest = dt.fread('../input/tabular-playground-series-oct-2021/test.csv').to_pandas().drop('id', axis=1)\ntest = reduce_memory_usage(test)\nss = dt.fread('../input/tabular-playground-series-oct-2021/sample_submission.csv').to_pandas()\nss = reduce_memory_usage(ss)\n")


# In[5]:


bool_cols_train = []
for i, col in enumerate(train.columns):
    if train[col].dtypes == bool:
        bool_cols_train.append(i)


# In[6]:


bool_cols_test = []
for i, col in enumerate(test.columns):
    if train[col].dtypes == bool:
        bool_cols_test.append(i)


# In[7]:


train.iloc[:, bool_cols_train] = train.iloc[:, bool_cols_train].astype(int)
test.iloc[:, bool_cols_test] = test.iloc[:, bool_cols_test].astype(int)


# In[8]:


print("Train set shape", train.shape, "\n", "Test set shape", test.shape)


# In[9]:


train.head()


# In[10]:


feature_cols = test.columns.tolist()

cnt_features =[]
cat_features =[]

for col in feature_cols:
    if train[col].dtype in ["float16", "float32", "float64"]:
        cnt_features.append(col)
    else:
        cat_features.append(col)
print(cat_features)


# In[11]:


X = train.drop('target', axis=1).copy()
y = train['target'].copy()
X_test = test.copy()

del train
gc.collect()
del test
gc.collect


# # Feature Engineering
# * Here I have target encoded the categorical features. Further I will update it with KFold target encoding.

# In[12]:


for cols in cat_features:
    enc = TargetEncoder(cols=[cols])
    X = enc.fit_transform(X, y)
    X_test = enc.transform(X_test)


# In[13]:


display(X.head())
display(X_test.head())


# # Model Training

# In[14]:


params = {
    'max_depth': 6,
    'n_estimators': 9500,
    'subsample': 0.7,
    'colsample_bytree': 0.2,
    'colsample_bylevel': 0.6000000000000001,
    'min_child_weight': 56.41980735551558,
    'reg_lambda': 75.56651890088857,
    'reg_alpha': 0.11766857055687065,
    'gamma': 0.6407823221122686,
    'booster': 'gbtree',
    'eval_metric': 'auc',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'use_label_encoder': False
    }


# In[15]:


get_ipython().run_cell_magic('time', '', 'kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=786)\n\npreds = []\nscores = []\n\nfor fold, (idx_train, idx_valid) in enumerate(kf.split(X, y)):\n    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]\n    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]\n    \n    params[\'learning_rate\']=0.007\n    model1 = XGBClassifier(**params)\n    \n    model1.fit(X_train,y_train,\n              eval_set=[(X_train, y_train),(X_valid,y_valid)],\n              early_stopping_rounds=200,\n              verbose=False)\n    \n    params[\'learning_rate\']=0.01\n    model2 = XGBClassifier(**params)\n    \n    model2.fit(X_train,y_train,\n              eval_set=[(X_train, y_train),(X_valid,y_valid)],\n              early_stopping_rounds=200,\n              verbose=False,\n              xgb_model=model1)\n    \n    params[\'learning_rate\']=0.05\n    model3 = XGBClassifier(**params)\n    \n    model3.fit(X_train,y_train,\n              eval_set=[(X_train, y_train),(X_valid,y_valid)],\n              early_stopping_rounds=200,\n              verbose=False,\n              xgb_model=model2)\n    \n    pred_valid = model3.predict_proba(X_valid)[:,1]\n    fpr, tpr, _ = roc_curve(y_valid, pred_valid)\n    score = auc(fpr, tpr)\n    scores.append(score)\n    \n    print(f"Fold: {fold + 1} Score: {score}")\n    print(\'||\'*40)\n    \n    test_preds = model3.predict_proba(X_test)[:,1]\n    preds.append(test_preds)\n    \nprint(f"Overall Validation Score: {np.mean(scores)}")\n')


# # Submission file

# In[16]:


predictions = np.mean(np.column_stack(preds),axis=1)

ss['target'] = predictions
ss.to_csv('./xgb.csv', index=False)
ss.head()

