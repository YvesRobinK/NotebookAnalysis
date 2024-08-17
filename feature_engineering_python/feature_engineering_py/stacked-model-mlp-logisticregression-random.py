#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# **Mainly based on @DES. 's notebook** [TPS08: LogisticRegression and some FE](https://www.kaggle.com/code/desalegngeb/tps08-logisticregression-and-some-fe)

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install feature-engine\n')


# In[2]:


import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Back, Style
from sklearn.preprocessing import StandardScaler
import itertools
from collections import defaultdict
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, HuberRegressor
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from feature_engine.encoding import WoEEncoder
from scipy.stats import spearmanr, rankdata
from tpot.builtins import StackingEstimator

np.random.seed(42)


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[3]:


train = pd.read_csv('../input/tabular-playground-series-aug-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-aug-2022/test.csv')
submission = pd.read_csv('../input/tabular-playground-series-aug-2022/sample_submission.csv')
print(f'train {train.shape}, test {test.shape}')


# In[4]:


def tweak_measurement_2(X:pd.DataFrame):
    return X.assign(
        measurement_2=X.groupby("product_code").measurement_2.transform(
            lambda s: s - s.mean()))


# **tweak_measurement_2** is from [A small exploitable anomaly](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/discussion/347222)

# In[5]:


target = train.pop('failure')

#train=tweak_measurement_2(train)
#test=tweak_measurement_2(test)

data = pd.concat([train, test])


# ## Feature Engineering

# In[6]:


def _scale(train_data, val_data, test_data, feats):
    scaler = StandardScaler()
       
    scaled_train = scaler.fit_transform(train_data[feats])
    scaled_val = scaler.transform(val_data[feats])
    scaled_test = scaler.transform(test_data[feats])
    
    #back to dataframe
    new_train = train_data.copy()
    new_val = val_data.copy()
    new_test = test_data.copy()
    
    new_train[feats] = scaled_train
    new_val[feats] = scaled_val
    new_test[feats] = scaled_test
    
    assert len(train_data) == len(new_train)
    assert len(val_data) == len(new_val)
    assert len(test_data) == len(new_test)
    
    return new_train, new_val, new_test


# In[7]:


def FeatureEngineering(data,train,test,target):
    data['m3_missing'] = data['measurement_3'].isnull().astype(np.int8)
    data['m5_missing'] = data['measurement_5'].isnull().astype(np.int8)
    data['area'] = data['attribute_2'] * data['attribute_3']

    feature = [f for f in test.columns if f.startswith('measurement') or f=='loading']

    # dictionary of dictionaries (for the 11 best correlated measurement columns), 
    # we will use the dictionaries below to select the best correlated columns according to the product code)
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
    col = [col for col in test.columns if 'measurement' not in col]+ ['loading','m3_missing','m5_missing']
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
    display(c.head(10))

    for i in range(10):
        measurement_col = 'measurement_' + c.iloc[i,0][12:] # we select the next best correlated column 
        fill_dict ={}
        for x in data.product_code.unique() : 
            corr = np.absolute(data[data.product_code == x].drop(col, axis=1).corr()[measurement_col]).sort_values(ascending=False)
            measurement_col_dic = {}
            measurement_col_dic[measurement_col] = corr[1:5].index.tolist()
            fill_dict[x] = measurement_col_dic[measurement_col]
        full_fill_dict[measurement_col] =fill_dict

    feature = [f for f in data.columns if f.startswith('measurement') or f=='loading']
    nullValue_cols = [col for col in train.columns if train[col].isnull().sum()!=0]

    for code in data.product_code.unique():
        total_na_filled_by_linear_model = 0
        for measurement_col in list(full_fill_dict.keys()):
            tmp = data[data.product_code==code]
            column = full_fill_dict[measurement_col][code]
            tmp_train = tmp[column+[measurement_col]].dropna(how='any')
            tmp_test = tmp[(tmp[column].isnull().sum(axis=1)==0)&(tmp[measurement_col].isnull())]

            model = HuberRegressor(epsilon=1.9)
            model.fit(tmp_train[column], tmp_train[measurement_col])
            data.loc[(data.product_code==code)&(data[column].isnull().sum(axis=1)==0)&(data[measurement_col].isnull()),measurement_col] = model.predict(tmp_test[column])
            total_na_filled_by_linear_model += len(tmp_test)

        # others NA columns:
        NA = data.loc[data["product_code"] == code,nullValue_cols ].isnull().sum().sum()
        model1 = KNNImputer(n_neighbors=3)
        #model1 = LGBMImputer(n_iter=50)
        #model1 = IterativeImputer(random_state=0) 
        data.loc[data.product_code==code, feature] = model1.fit_transform(data.loc[data.product_code==code, feature])

    data['measurement_avg'] = data[[f'measurement_{i}' for i in range(3, 17)]].mean(axis=1)
    
    train = data.iloc[:train.shape[0],:]
    test = data.iloc[train.shape[0]:,:]

    groups = train.product_code
    X = train
    y = target
    
    woe_encoder = WoEEncoder(variables=['attribute_0'])
    woe_encoder.fit(train, y)
    X = woe_encoder.transform(train)
    test = woe_encoder.transform(test)
    
    X['measurement(3*5)'] = X['measurement_3'] * X['measurement_5']
    test['measurement(3*5)'] = test['measurement_3'] * test['measurement_5']

    X['missing(3*5)'] = X['m5_missing'] * (X['m3_missing'])
    test['missing(3*5)'] = test['m5_missing'] * (test['m3_missing'])
    
    return X,y,test


# In[8]:


X,y,test=FeatureEngineering(data,train,test,target)


# In[ ]:





# In[9]:


select_feature = [
    'loading',
    'attribute_0',
    'measurement_17',
    'measurement_0',
    'measurement_1',
    'measurement_2',
    'area',
    'm3_missing', 
    'm5_missing',
    'measurement_avg',
    'measurement(3*5)',
    'missing(3*5)',      
]


# ## Model

# In[10]:


seeds = [4, 8, 12, 24, 48, 64, 86, 128, 256]


lr_oof = np.zeros(len(train))
lr_test = np.zeros(len(test))
lr_auc = 0

for i, seed in enumerate(seeds):
    print('------- Seed:%2d -------' % (seeds[i]))
    np.random.seed(seed)
    #kf = GroupKFold(n_splits=5)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        x_train, x_val, x_test = _scale(x_train, x_val, test, select_feature)

        '''
        model = make_pipeline(
                    StackingEstimator(estimator=LogisticRegression(max_iter=500,C=1000, dual=False, penalty="l2",solver='newton-cg')),
                    #Normalizer(norm="l1"),
                    StackingEstimator(
                        estimator=MLPClassifier(alpha=3.911908347551938e-05, learning_rate_init=0.9991485969980523)
                    ),
                    LogisticRegression(max_iter=500,C=1000, dual=False, penalty="l2",solver='newton-cg')
                )
        '''
        model = make_pipeline(
                    StackingEstimator(estimator=LogisticRegression(max_iter=500,C=1000, dual=False, penalty="l2",solver='newton-cg')),
                    #Normalizer(norm="l1"),
                    StackingEstimator(
                        estimator=MLPClassifier(alpha=0.0002042624861478862, learning_rate_init=0.14551105194911998)
                    ),
                    LogisticRegression(max_iter=500,C=20, dual=False, penalty="l2",solver='newton-cg')
                )
        '''
        model = make_pipeline(
                    StackingEstimator(estimator=LogisticRegression(max_iter=500,C=0.002, dual=False, penalty="l2",solver='newton-cg')),
                    #Normalizer(norm="l1"),
                    StackingEstimator(
                        estimator=MLPClassifier(alpha=2.646953481533627e-05, learning_rate_init=0.08958560094341549)
                    ),
                    LogisticRegression(max_iter=500,C=1000.0, dual=False, penalty="l2",solver='newton-cg')
                )
        '''
        model.fit(x_train[select_feature], y_train)

        val_preds = model.predict_proba(x_val[select_feature])[:, 1]
        #print("FOLD: ", fold_idx+1, " ROC-AUC:", round(roc_auc_score(y_val, val_preds), 5))
        lr_auc += roc_auc_score(y_val, val_preds) / (5*len(seeds))
        lr_test += model.predict_proba(x_test[select_feature])[:, 1] /  len(seeds)
        lr_oof[val_idx] = val_preds

print(f"{Fore.GREEN}{Style.BRIGHT}Average auc = {round(lr_auc, 5)}{Style.RESET_ALL}")
print(f"{Fore.BLUE}{Style.BRIGHT}OOF auc = {round(roc_auc_score(y, lr_oof), 5)}{Style.RESET_ALL}")
print(f"{Fore.RED}{Style.BRIGHT}Public LB = {0.59098}{Style.RESET_ALL}")


# 
# ---
# 
# The original notebook's results are:
# 
# **Average auc = 0.59021**
# 
# **OOF auc = 0.59007**
# 
# **Public LB = 0.59118**
# 
# We have higher CV score and higher public LB score.

# In[11]:


submission['failure'] = lr_test
submission.to_csv("./submission.csv", index=False)


# In[12]:


sns.histplot(x=submission['failure'])
print('submission data failure distribution')


# In[ ]:




