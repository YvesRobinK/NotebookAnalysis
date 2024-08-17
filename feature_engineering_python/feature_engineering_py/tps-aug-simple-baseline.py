#!/usr/bin/env python
# coding: utf-8

# ### TPS-AUG22
# #### Simple Baseline
# _____
# 
# On this notebook we train a simple baseline model for this competition. 
# The goal of this notebook is being a starter notebook when approaching this competition. It will host some of the high scoring discoveries on the early stage of the competition and will be constantly updated throughout the beggining of the competition.
# 
# #### The Current Version:
# 
# **We start with missing values imputation:**
# - We split the imputation process into two different phases: 
#      - For columns that are highly correlated with the target column - we use a linear model to estimate the missing values (HuberRegressor).
#      - For all other columns we use KNN inputer. 
# 
# **We then continue with simple feature engineering:**
# - We add the `measurement_avg` which is the avg of all the measurement columns.
# - We then extract the `WoEEncoder` feature that was introduced by [maxsarmento](https://www.kaggle.com/MAXSARMENTO).
# 
# **During the training loop we scale the data:**
# - Using simple `StandardScaler` 
# 
# **And for modeling**
# - We use grouped stratified KFold 
# - With LogisticRegression
# - And HuberRegressor
# - And LGBMClassifier
# 
# 
# 

# **Installations (Hidden Cell)**

# In[1]:


get_ipython().system('pip install feature_engine')
get_ipython().system('git clone https://github.com/analokmaus/kuma_utils.git')
import sys; sys.path.append("kuma_utils/")


# #### Imports

# In[2]:


import os
import sys
import joblib
import numpy as np
import pandas as pd
import gc; gc.enable()
from lightgbm import LGBMClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from feature_engine.encoding import WoEEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from kuma_utils.preprocessing.imputer import LGBMImputer
from sklearn.linear_model import LogisticRegression, HuberRegressor
import warnings; warnings.filterwarnings("ignore")


# #### Data Loading

# In[3]:


df_train = pd.read_csv("/kaggle/input/tabular-playground-series-aug-2022/train.csv")
df_test = pd.read_csv("/kaggle/input/tabular-playground-series-aug-2022/test.csv")
sub = pd.read_csv("/kaggle/input/tabular-playground-series-aug-2022/sample_submission.csv")
target, groups = df_train['failure'], df_train['product_code']
df_train.drop('failure',axis=1, inplace = True)


# #### Preprocessing
# 
# > **Credit:** This clean minimalist function comes from [this](https://www.kaggle.com/code/pourchot/hunting-for-missing-values) great notebook by [Laurent Pourchot](https://www.kaggle.com/pourchot)

# In[4]:


def preprocessing(df_train, df_test):
    data = pd.concat([df_train, df_test])
    
    data['m3_missing'] = data['measurement_3'].isnull().astype(np.int8)
    data['m5_missing'] = data['measurement_5'].isnull().astype(np.int8)
    data['area'] = data['attribute_2'] * data['attribute_3']

    feature = [f for f in df_test.columns if f.startswith('measurement') or f=='loading']

    # dictionnary of dictionnaries (for the 11 best correlated measurement columns), 
    # we will use the dictionnaries below to select the best correlated columns according to the product code)
    # Only for 'measurement_17' we make a 'manual' selection :
    full_fill_dict ={}
    full_fill_dict['measurement_17'] = {
        'A': ['measurement_5','measurement_6','measurement_8'],
        'B': ['measurement_4','measurement_5','measurement_7'],
        'C': ['measurement_5','measurement_7','measurement_8','measurement_9'],
        'D': ['measurement_5','measurement_6','measurement_7','measurement_8'],
        'E': ['measurement_4','measurement_5','measurement_6','measurement_8'],
        'F': ['measurement_4','measurement_5','measurement_6','measurement_7'],
        'G': ['measurement_4','measurement_6','measurement_8','measurement_9'],
        'H': ['measurement_4','measurement_5','measurement_7','measurement_8','measurement_9'],
        'I': ['measurement_3','measurement_7','measurement_8']
    }

    # collect the name of the next 10 best measurement columns sorted by correlation (except 17 already done above):
    col = [col for col in df_test.columns if 'measurement' not in col]+ ['loading','m3_missing','m5_missing']
    a = []
    b =[]
    for x in range(3,17):
        corr = np.absolute(data.drop(col, axis=1).corr()[f'measurement_{x}']).sort_values(ascending=False)
        a.append(np.round(np.sum(corr[1:4]),3)) # we add the 3 first lines of the correlation values to get the "most correlated"
        b.append(f'measurement_{x}')
    c = pd.DataFrame()
    c['Selected columns'] = b
    c['correlation total'] = a
    c = c.sort_values(by = 'correlation total',ascending=False).reset_index(drop = True)
    print(f'Columns selected by correlation sum of the 3 first rows : ')
    display(c.head(10))

    for i in range(10):
        measurement_col = 'measurement_' + c.iloc[i,0][12:] # we select the next best correlated column 
        fill_dict = {}
        for x in data.product_code.unique() : 
            corr = np.absolute(data[data.product_code == x].drop(col, axis=1).corr()[measurement_col]).sort_values(ascending=False)
            measurement_col_dic = {}
            measurement_col_dic[measurement_col] = corr[1:5].index.tolist()
            fill_dict[x] = measurement_col_dic[measurement_col]
        full_fill_dict[measurement_col] =fill_dict

    feature = [f for f in data.columns if f.startswith('measurement') or f=='loading']
    nullValue_cols = [col for col in df_train.columns if df_train[col].isnull().sum()!=0]

    for code in data.product_code.unique():
        total_na_filled_by_linear_model = 0
        print(f'\n-------- Product code {code} ----------\n')
        print(f'filled by linear model :')
        for measurement_col in list(full_fill_dict.keys()):
            tmp = data[data.product_code == code]
            column = full_fill_dict[measurement_col][code]
            tmp_train = tmp[column+[measurement_col]].dropna(how='any')
            tmp_test = tmp[(tmp[column].isnull().sum(axis=1)==0)&(tmp[measurement_col].isnull())]

            model = HuberRegressor(epsilon=1.9)
            model.fit(tmp_train[column], tmp_train[measurement_col])
            data.loc[(data.product_code==code)&(data[column].isnull().sum(axis=1)==0)&(data[measurement_col].isnull()),measurement_col] = model.predict(tmp_test[column])
            print(f'{measurement_col} : {len(tmp_test)}')
            total_na_filled_by_linear_model += len(tmp_test)

        # others NA columns:
        NA = data.loc[data["product_code"] == code,nullValue_cols ].isnull().sum().sum()
        model1 = KNNImputer(n_neighbors=3)
        data.loc[data.product_code==code, feature] = model1.fit_transform(data.loc[data.product_code==code, feature])
        print(f'\n{total_na_filled_by_linear_model} filled by linear model ') 
        print(f'{NA} filled by KNN ')

    data['measurement_avg'] = data[[f'measurement_{i}' for i in range(3, 17)]].mean(axis=1)
    df_train = data.iloc[:df_train.shape[0],:]
    df_test = data.iloc[df_train.shape[0]:,:]

    woe_encoder = WoEEncoder(variables=['attribute_0'])
    woe_encoder.fit(df_train, target)
    df_train = woe_encoder.transform(df_train)
    df_test = woe_encoder.transform(df_test)

    features = ['loading', 'attribute_0', 'measurement_17', 'measurement_0', 'measurement_1', 'measurement_2', 'area', 'm3_missing', 'm5_missing', 'measurement_avg']
    
    return df_train, df_test, features

def scale(train_data, val_data, test_data, feats):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data[feats])
    scaled_val = scaler.transform(val_data[feats])
    scaled_test = scaler.transform(test_data[feats])
    new_train = train_data.copy()
    new_val = val_data.copy()
    new_test = test_data.copy()
    new_train[feats] = scaled_train
    new_val[feats] = scaled_val
    new_test[feats] = scaled_test
    return new_train, new_val, new_test

df_train, df_test, features = preprocessing(df_train, df_test)
df_train['failure'] = target


# #### Modeling

# In[5]:


df_train[features]


# **No splits submission**

# In[6]:


output = pd.read_csv('../input/tabular-playground-series-aug-2022/sample_submission.csv')

x_train, x_val, x_test = scale(df_train[features], df_train[features], df_test[features], features)

model = LogisticRegression(max_iter=200, C=0.0001, penalty='l2', solver='newton-cg')
model.fit(x_train, target)
output['failure'] = (model.predict_proba(x_test)[:, 1]) * 0.8

model = LGBMClassifier(**{'seed': 42, 'n_jobs': -1, 'lambda_l2': 2, 'metric': "auc", 'max_depth': -1, 'num_leaves': 100, 'boosting': 'gbdt', 'bagging_freq': 10, 'learning_rate': 0.01, 'objective': 'binary', 'min_data_in_leaf': 40, 'num_boost_round': 70, 'feature_fraction': 0.90, 'bagging_fraction': 0.90})
model.fit(x_train, target)
output['failure'] += (model.predict_proba(x_test)[:, 1]) * 0.2

output.to_csv('submission_no_splits.csv', index=False)


# **CV submission**

# In[7]:


params = {"max_iter": 200, "C": 0.0001, "penalty": "l2", "solver": "newton-cg"}

oof = np.zeros(len(df_train))
test_preds = np.zeros(len(df_test))
for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0).split(df_train, df_train["failure"])):
    x_train, y_train = df_train.loc[train_idx][features], df_train.loc[train_idx]["failure"]
    x_val, y_val = df_train.loc[val_idx][features], df_train.loc[val_idx]["failure"]

    x_train, x_val, x_test = scale(x_train, x_val, df_test, features)
    
    model = LogisticRegression(**params)
    model.fit(x_train, y_train)
    y_pred_1 = model.predict_proba(x_val)[:, 1]
    test_preds_1 = model.predict_proba(df_test[features])[:, 1] / 5
       
    lgb_params = {
        'seed': 42,
        'n_jobs': -1,
        'lambda_l2': 2,
        'metric': "auc",
        'max_depth': -1,
        'num_leaves': 100,
        'boosting': 'gbdt',
        'bagging_freq': 10,
        'learning_rate': 0.01,
        'objective': 'binary',
        'min_data_in_leaf': 40,
        'num_boost_round': 1000,
        'feature_fraction': 0.90,
        'bagging_fraction': 0.90,
    }
    
    model = LGBMClassifier(**lgb_params)
    model.fit(x_train, y_train, eval_set = [(x_val, y_val)], early_stopping_rounds = 30)            
    y_pred_2 = model.predict_proba(x_val)[:, 1]
    test_preds_2 = model.predict_proba(df_test[features])[:, 1] / 5
        
    model = GaussianNB(var_smoothing=0.5, priors=[len(y_train[y_train == 0]) / len(y_train), len(y_train[y_train == 1])/len(y_train)])
    model.fit(x_train, y_train)
    y_pred_3 = model.predict_proba(x_val)[:, 1]
    test_preds_3 = model.predict_proba(x_test[features])[:, 1] / 5
    
    oof[val_idx] = (y_pred_1 * 1.0)     + (0.0 * y_pred_2) + (0.0 * y_pred_3)
    test_preds   = (test_preds_1 * 1.0) + (0.0 * test_preds_2) + (0.0 * test_preds_3)
    
    print(f"Val score: {roc_auc_score(y_val, oof[val_idx]):.7f}")

print(f"Val score: {roc_auc_score(df_train['failure'], oof):.7f}")


# #### Submission

# In[8]:


output = pd.read_csv('../input/tabular-playground-series-aug-2022/sample_submission.csv')
output['failure'] = test_preds
output.to_csv('submission.csv', index=False)

