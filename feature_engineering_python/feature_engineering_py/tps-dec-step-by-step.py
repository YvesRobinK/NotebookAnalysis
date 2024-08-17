#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import math
import gc
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy import stats
from plotly.subplots import make_subplots
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('/kaggle/input/tabular-playground-series-dec-2021/train.csv')
test = pd.read_csv('/kaggle/input/tabular-playground-series-dec-2021/test.csv')

cols = [e for e in test.columns if e not in ('Id')]
continous_features = cols[:10]
categorical_features = cols[10:]


# # Exploratory Data Analysis

# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


train.info()


# In[6]:


test.info()


# In[7]:


#Check if there'is null values
train.isnull().sum()


# In[8]:


#Check if there'is null values
test.isnull().sum()


# * No Null values 😀

# In[9]:


train[continous_features].describe()


# In[10]:


test[continous_features].describe()


# In[11]:


# plot continous features 
i = 1
plt.figure()
fig, ax = plt.subplots(2, 5,figsize=(20, 12))
for feature in continous_features:
    plt.subplot(2, 5,i)
    sns.histplot(train[feature],color="blue", kde=True,bins=100, label='train_'+feature)
    sns.histplot(test[feature],color="olive", kde=True,bins=100, label='test_'+feature)
    plt.xlabel(feature, fontsize=9); plt.legend()
    i += 1
plt.show() 


# ## Target distibution

# In[12]:


sns.catplot(x="Cover_Type", kind="count", palette="ch:.25", data=train)


# In[13]:


train.Cover_Type.value_counts()


# * the data is unbalanced 😕
# * we have only one sample with target 5 !!! 😰 

# ## Features correlation(I will use only Continous Features)

# In[14]:


corr = train[continous_features+['Cover_Type']].corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(3)


# ## let's reduce the memory usage

# In[15]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_memory = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_memory = df.memory_usage().sum() / 1024**2
    if verbose: 
        print(f"Memory usage of dataframe after reduction {end_memory} MB")
        print(f"Reduced by {100 * (start_memory - end_memory) / start_memory} % ")
    return df


# In[16]:


train[cols] = reduce_mem_usage(train[cols])


# In[17]:


test[cols] = reduce_mem_usage(test[cols])


# * We reduced the train dataset from 1.7 GB to 240 MB 
# * We reduced the test dataset from 419 MB to 60 MB

# ## Feature Engineering

# In[18]:


# delete the sample with target 5
train.drop(train[train['Cover_Type']==5].index,inplace=True)


# In[19]:


# generate new features 
cols = [e for e in test.columns if e not in ('Id')]

train['binned_elevation'] = [math.floor(v/50.0) for v in train['Elevation']]
test['binned_elevation'] = [math.floor(v/50.0) for v in test['Elevation']]

train['Horizontal_Distance_To_Roadways_Log'] = [np.log(v+300) for v in train['Horizontal_Distance_To_Roadways']]
test['Horizontal_Distance_To_Roadways_Log'] = [np.log(v+300) for v in test['Horizontal_Distance_To_Roadways']]

train['Soil_Type12_32'] = train['Soil_Type32'] + train['Soil_Type12']
test['Soil_Type12_32'] = test['Soil_Type32'] + test['Soil_Type12']
train['Soil_Type23_22_32_33'] = train['Soil_Type23'] + train['Soil_Type22'] + train['Soil_Type32'] + train['Soil_Type33']
test['Soil_Type23_22_32_33'] = test['Soil_Type23'] + test['Soil_Type22'] + test['Soil_Type32'] + test['Soil_Type33']

cols = [e for e in test.columns if e not in ('Id')]


# In[20]:


scaler = StandardScaler()
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.transform(test[cols])


# # Let's build a lightgbm model

# In[21]:


# I optained these parameters using OPTUNA
# check this kernel to learn more about OPTUNA : https://www.kaggle.com/hamzaghanmi/lgbm-hyperparameter-tuning-using-optuna
params = {'objective': 'multiclass',  'random_state': 48,'n_estimators': 20000,
            'n_jobs': -1,'reg_alpha': 0.9481920810028138, 'reg_lambda': 8.15049828410672, 'colsample_bytree': 0.5, 'subsample': 0.8,
          'learning_rate': 0.2, 'max_depth': 100, 'num_leaves': 26, 'min_child_samples': 88, 'cat_smooth': 78}


# In[22]:


preds = [] 
kf = StratifiedKFold(n_splits=10,random_state=48,shuffle=True)
acc=[]  # list contains accuracy for each fold
n=0
for trn_idx, test_idx in kf.split(train[cols],train['Cover_Type']):
    X_tr,X_val = train[cols].iloc[trn_idx],train[cols].iloc[test_idx]
    y_tr,y_val = train['Cover_Type'].iloc[trn_idx],train['Cover_Type'].iloc[test_idx]
    
    model = LGBMClassifier(**params)
    model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],early_stopping_rounds=100,verbose=False)
    
    preds.append(model.predict(test[cols]))
    acc.append(accuracy_score(y_val, model.predict(X_val)))

    print(f"fold: {n+1} , accuracy: {round(acc[n]*100,3)}")  
    n+=1  
    
    del X_tr,X_val,y_tr,y_val
    gc.collect()     


# In[23]:


print(f"the mean Accuracy is : {round(np.mean(acc)*100,3)} ")


# In[24]:


# most 30 important features for lgb model
from optuna.integration import lightgbm as lgb
lgb.plot_importance(model, max_num_features=30, figsize=(10,10))
plt.show() 


# ## Let's Make a Submission

# In[25]:


preds


# * Now we have 10 arrays and each array was calculated for the i-th fold (we used 10 folds).<br>
# * So I'm going to use the <b>mode</b> in order to generate the final prediction.<br>
# <img src="https://k8schoollessons.com/wp-content/uploads/2019/06/Median-Mode-Mean-and-Range-1.jpg" width="450px"/>

# In[26]:


sub = pd.read_csv('/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv')
prediction = stats.mode(preds)[0][0]
sub['Cover_Type'] = prediction
sub.to_csv('submission.csv', index=False)


# In[27]:


sub


# # I hope that you find this kernel usefull🏄
