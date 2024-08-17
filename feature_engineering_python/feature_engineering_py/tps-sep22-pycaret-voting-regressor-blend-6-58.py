#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:#c81d25;">Table of Contents</h1>

#  | S.No       |                   Heading                |
#  | :------------- | :-------------------:                |                     
#  |  01 |  [**Libraries**](#library)                        |  
#  |  02 |  [**Read Data**](#read_data)                |
#  |  03 |  [**Missing values**](#missing)      |
#  |  04 |  [**Feature engineering with date attribute**](#fe)                |
#  |  05 |  [**Define SMAPE metric**](#smape)  |
#  |  06 |  [**PyCaret Regression Setup**](#pycaret)   |
#  |  07 |  [**Train & Compare models**](#train)   |
#  |  08 |  [**Blend Models : Voting Regressor**](#blend) |
#  |  09 |  [**Catboost Feature Importance**](#catboost) |
#  |  10 |  [**LGBM Feature Importance**](#lgbm) |
#  |  11 |  [**XgBoost Feature Importance**](#xgboost) |
#  |  12 |  [**Residual Plots**](#residual) |
#  |  13 |  [**Train on Entire Data**](#entire) |
#  |  14 |  [**Generate Test Predictions**](#generate) |
# 

# <a id="library"></a>
# <h1 style="color:#c81d25;">Libraries</h1>

# In[3]:


get_ipython().run_cell_magic('capture', '', '!pip install pycaret[full] --ignore-installed llvmlite\nimport pandas as pd\nfrom pycaret.regression import *\n')


# In[4]:


import numpy as np


# <a id='read_data' ></a>
# <h1 style="color:#c81d25;">Read Data</h1>

# In[5]:


df_train = pd.read_csv('../input/tabular-playground-series-sep-2022/train.csv', index_col = 'row_id')
df_test = pd.read_csv('../input/tabular-playground-series-sep-2022/test.csv', index_col = 'row_id')


# In[ ]:


df_train.head()


# <a id='missing' ></a>
# <h1 style="color:#c81d25;">Missing values</h1>

# In[6]:


df_train.isna().sum()


# In[7]:


df_test.isna().sum()


# <a id='fe' ></a>
# <h1 style="color:#c81d25;">Feature engineering with date attribute</h1>

# In[8]:


def get_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['week']= df['date'].dt.week
    df['year'] = 'Y' + df['date'].dt.year.astype(str)
    df['quarter'] = 'Q' + df['date'].dt.quarter.astype(str)
    df['day'] = df['date'].dt.day
    df['dayofyear'] = df['date'].dt.dayofyear
    df.loc[(df.date.dt.is_leap_year) & (df.dayofyear >= 60),'dayofyear'] -= 1
    df['weekend'] = df['date'].dt.weekday >=5
    df['weekday'] = 'WD' + df['date'].dt.weekday.astype(str)
    df.drop(columns=['date'],inplace=True)  

get_features(df_train)
get_features(df_test)


# <a id='smape' ></a>
# <h1 style="color:#c81d25;">Define SMAPE metric</h1>

# In[9]:


def SMAPE(y_true, y_pred):
    denominator = (y_true + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)


# <a id='pycaret' ></a>
# <h1 style="color:#c81d25;">Pycaret Regression setup</h1>

# In[11]:


experiment = setup(data = df_train,
            target = 'num_sold',
            train_size = 0.75,
            normalize = True, 
            normalize_method = 'robust', 
            transform_target = True,
            transformation=True,
            data_split_shuffle = False, 
            feature_interaction = True, 
            use_gpu = True, 
            silent = True,
            fold = 20,
            n_jobs = -1)


# In[12]:


add_metric('SMAPE', 'SMAPE', SMAPE, greater_is_better = False)


# <a id='train' ></a>
# <h1 style="color:#c81d25;">Train & Compare models</h1>

# In[13]:


best_models = compare_models(include=['xgboost','catboost','lightgbm'],sort = 'SMAPE',n_select=3)


# <a id='blend' ></a>
# <h1 style="color:#c81d25;">Blend models : Voting Regressor</h1>

# In[15]:


blend = blend_models(best_models,choose_better=True,optimize='SMAPE')
predict_model(blend);


# <a id='catboost' ></a>
# <h2 style="color:#c81d25;">Catboost Feature Importance</h2>

# In[22]:


plot_model(blend.estimators_[0], plot = 'feature', use_train_data = True)


# <a id='lgbm' ></a>
# <h2 style="color:#c81d25;">LGBM Feature Importance</h2>

# In[23]:


plot_model(blend.estimators_[1], plot = 'feature', use_train_data = True)


# <a id='xgboost' ></a>
# <h2 style="color:#c81d25;">XGBoost Feature Importance</h2>

# In[24]:


plot_model(blend.estimators_[2], plot = 'feature', use_train_data = True)


# <a id='residual' ></a>
# <h1 style="color:#c81d25;">Residual Plots</h1>

# In[25]:


plot_model(blend.estimators_[0], plot = 'residuals', use_train_data = False)


# In[26]:


plot_model(blend.estimators_[1], plot = 'residuals', use_train_data = False)


# In[27]:


plot_model(blend.estimators_[2], plot = 'residuals', use_train_data = False)


# <a id="entire"></a>
# <h1 style="color:#c81d25;">Train on Entire data</h1>

# In[16]:


final_blend_model = finalize_model(blend)
predict_model(final_blend_model);


# <a id="generate"></a>
# 
# <h1 style="color:#c81d25;">Generate Test Predictions</h1>

# In[18]:


test_predictions= predict_model(final_blend_model, data=df_test)
test_predictions.head()


# In[20]:


submission = pd.DataFrame(list(zip(df_test.index, test_predictions.Label)),columns = ['row_id', 'num_sold'])

submission.to_csv('submission_final.csv', index = False)

