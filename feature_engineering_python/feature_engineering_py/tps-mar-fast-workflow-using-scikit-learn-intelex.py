#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:rgba(0, 167, 255, 0.6);border-radius:5px;display:fill">
#     <h1><center>Tabular Playground Series - Mar 2022
# </div>

# <center><a><img src="https://i.ibb.co/PWvpT9F/header.png" alt="header" border="0" width=800 height=400 class="center"></a>

# <h1> Fast Random Forest and Intel® Extension for Scikit-learn* - Kaggle Tabular Playground Series - March 2022 </h1>

# For classical machine learning algorithms, we often use the most popular Python library, Scikit-learn. With Scikit-learn you can fit models and search for optimal parameters, but it sometimes works for hours. Speeding up this process is something anyone who uses Scikit-learn would be interested in.
# 
# I want to show you how to use Scikit-learn library and get the results faster without changing the code. To do this, we will make use of another Python library, [**Intel® Extension for Scikit-learn***](https://github.com/intel/scikit-learn-intelex). It accelerates Scikit-learn and does not require you to change the code written for Scikit-learn.
# 
# I will show you how to **speed up** your kernel without changing your code!

# More information you can find in [Introduction to scikit-learn-intelex](https://www.kaggle.com/lordozvlad/introduction-to-scikit-learn-intelex)!

# ### Intel® Extension for Scikit-learn installation:

# In[1]:


get_ipython().system('pip install scikit-learn-intelex -q --progress-bar off')


# ### Import Libraries

# In[2]:


import pandas as pd
import numpy as np
import warnings
import gc
from IPython.display import HTML
warnings.filterwarnings("ignore")

from math import sin, cos, pi

from timeit import default_timer as timer
import matplotlib.pyplot as plt

random_state = 42


# In[3]:


from sklearnex import patch_sklearn
patch_sklearn()


# ### Reading Data

# In[4]:


PATH_TRAIN      = '../input/tabular-playground-series-mar-2022/train.csv'
PATH_TEST       = '../input/tabular-playground-series-mar-2022/test.csv'
PATH_SUBMISSION = '../input/tabular-playground-series-mar-2022/sample_submission.csv'


# In[5]:


train_data = pd.read_csv(PATH_TRAIN, parse_dates=['time'])
test_data  = pd.read_csv(PATH_TEST, parse_dates=['time'])
submission = pd.read_csv(PATH_SUBMISSION)


# In[6]:


train_data[:5]


# <a id="top"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" role="tab" aria-controls="home">Feature engineering</h3>
#     
#    * [Date-Related Features](#1)
#    * [Lag Features](#2)
#    * [Expanding Window Feature](#3)
#    * [Cyclical Features](#4)
#    

# <a id="1"></a>
# ### Date-Related Features
# <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Go to Feature engineering</a>

# In[7]:


train_data['year'] = train_data['time'].dt.year 
train_data['month'] = train_data['time'].dt.month 
train_data['day'] = train_data['time'].dt.day
train_data['hour'] = train_data['time'].dt.hour
train_data['minute'] = train_data['time'].dt.minute
train_data['weekday'] = train_data['time'].dt.weekday
train_data['dayofweek_num'] = train_data['time'].dt.dayofweek  

test_data['year'] = test_data['time'].dt.year 
test_data['month'] = test_data['time'].dt.month 
test_data['day'] = test_data['time'].dt.day
test_data['hour'] = test_data['time'].dt.hour
test_data['minute'] = test_data['time'].dt.minute
test_data['weekday'] = test_data['time'].dt.weekday
test_data['dayofweek_num'] = test_data['time'].dt.dayofweek  


# In[8]:


train_data[:5]


# <a id="2"></a>
# ### Lag Features
# <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Go to Feature engineering</a>

# In[9]:


train_data['lag_1'] = train_data['congestion'].shift(1)
train_data['lag_2'] = train_data['congestion'].shift(2)
train_data['lag_3'] = train_data['congestion'].shift(3)
train_data['lag_4'] = train_data['congestion'].shift(4)
train_data['lag_5'] = train_data['congestion'].shift(5)
train_data['lag_6'] = train_data['congestion'].shift(6)
train_data['lag_7'] = train_data['congestion'].shift(7)


# In[10]:


train_data[:5]


# <a id="3"></a>
# ### Expanding Window Feature
# <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Go to Feature engineering</a>

# In[11]:


train_data['expanding_mean'] = train_data['congestion'].expanding(2).mean()


# In[12]:


train_data[:5]


# <a id="4"></a>
# ### Cyclical Features
# <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Go to Feature engineering</a>

# For this features engineering techniques thank you [Inversion](https://www.kaggle.com/inversion) and his wonderfull [notebook](https://www.kaggle.com/inversion/tps-mar-22-cyclical-features).

# In[13]:


sin_vals = {
    'NB': 0.0,
    'NE': sin(1 * pi/4),
    'EB': 1.0,
    'SE': sin(3 * pi/4),
    'SB': 0.0,
    'SW': sin(5 * pi/4),    
    'WB': -1.0,    
    'NW': sin(7 * pi/4),  
}

cos_vals = {
    'NB': 1.0,
    'NE': cos(1 * pi/4),
    'EB': 0.0,
    'SE': cos(3 * pi/4),
    'SB': -1.0,
    'SW': cos(5 * pi/4),    
    'WB': 0.0,    
    'NW': cos(7 * pi/4),  
}


train_data['sin'] = train_data['direction'].map(sin_vals)
test_data['sin'] = test_data['direction'].map(sin_vals)

train_data['cos'] = train_data['direction'].map(cos_vals)
test_data['cos'] = test_data['direction'].map(cos_vals)


# In[14]:


encoded_vals = {
    'NB': 0,
    'NE': 1,
    'EB': 2,
    'SE': 3,
    'SB': 4,
    'SW': 5,
    'WB': 6, 
    'NW': 7,
}

train_data['direction'] = train_data['direction'].map(encoded_vals)
test_data['direction'] = test_data['direction'].map(encoded_vals)


# In[15]:


train_data['hour_sin'] = np.sin(2 * np.pi * train_data['hour']/23.0)
train_data['hour_cos'] = np.cos(2 * np.pi * train_data['hour']/23.0)
train_data['minute_sin'] = np.sin(2 * np.pi * train_data['minute']/59.0)
train_data['minute_cos'] = np.cos(2 * np.pi * train_data['minute']/59.0)

test_data['hour_sin'] = np.sin(2 * np.pi * test_data['hour']/23.0)
test_data['hour_cos'] = np.cos(2 * np.pi * test_data['hour']/23.0)
test_data['minute_sin'] = np.sin(2 * np.pi * test_data['minute']/59.0)
test_data['minute_cos'] = np.cos(2 * np.pi * test_data['minute']/59.0)


# In[16]:


train_data[:5]


# In[17]:


train_data = train_data.fillna(0)


# In[18]:


train_data = train_data.drop('time', axis='columns')
test_data = test_data.drop('time', axis='columns')


# In[19]:


X, y = train_data.drop(['congestion'], axis = 1), train_data['congestion']


# In[20]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)


# ### ELI5

# ELI5 provides a way to compute feature importances for any black-box estimator by measuring how score decreases when a feature is not available.

# In[21]:


import eli5
from eli5.sklearn import PermutationImportance


# In[22]:


from sklearn.ensemble import RandomForestRegressor


# In[23]:


timeFirstI  = timer()
modelRF     = RandomForestRegressor(n_estimators = 100, random_state = 42, max_depth = 5, n_jobs = -1).fit(X_train, y_train)
perm        = PermutationImportance(modelRF, random_state = 42).fit(X_val, y_val)
timeSecondI = timer()


# In[24]:


print("Total time with Intel Extension: {} seconds".format(timeSecondI - timeFirstI))


# In[25]:


eli5.show_weights(perm, feature_names = X.columns.tolist())


# In[26]:


pi_features = eli5.explain_weights_df(perm, feature_names = X_train.columns.tolist())
pi_features = pi_features.loc[pi_features['weight'] >= 0.005]['feature'].tolist()


# In[27]:


pi_features


# In[28]:


X_trains = X_train.loc[:, pi_features]


# In[29]:


train_features = []
for features in pi_features:
    if features[:3] != 'lag' and features != 'expanding_mean':
        train_features.append(features)

train_features


# In[30]:


X_trainRF = X_train.loc[:, train_features]


# In[31]:


test_features = []
for features in pi_features:
    if features[:3] != 'lag' and features != 'expanding_mean':
        test_features.append(features)

test_features


# In[32]:


test_data = test_data.loc[:, test_features]


# ### Default Scikit-learn

# In[33]:


from sklearnex import unpatch_sklearn
unpatch_sklearn()


# In[34]:


import eli5
from eli5.sklearn import PermutationImportance


# In[35]:


from sklearn.ensemble import RandomForestRegressor


# In[36]:


timeFirstD  = timer()
modelRF     = RandomForestRegressor(n_estimators = 100, random_state = 42, max_depth = 5, n_jobs = -1).fit(X_train, y_train)
perm        = PermutationImportance(modelRF, random_state = 42).fit(X_val, y_val)
timeSecondD = timer()


# In[37]:


print("Total time with default Scikit-learn: {} seconds".format(timeSecondD - timeFirstD))


# In[38]:


eli5.show_weights(perm, feature_names = X.columns.tolist())


# In[39]:


pi_features = eli5.explain_weights_df(perm, feature_names = X_train.columns.tolist())
pi_features = pi_features.loc[pi_features['weight'] >= 0.005]['feature'].tolist()


# In[40]:


pi_features


# In[41]:


eli5_speedup = round((timeSecondD - timeFirstD) / (timeSecondI - timeFirstI), 2)
HTML(f'<h2>ELI5 speedup: {eli5_speedup}x</h2>'
     f'(from {round((timeSecondD - timeFirstD), 2)} to {round((timeSecondI - timeFirstI), 2)} seconds)')


# ### Build model

# ### Optimized Scikit-learn

# In[42]:


from sklearnex import patch_sklearn
patch_sklearn()


# In[43]:


from sklearn.ensemble import RandomForestRegressor


# In[44]:


rf = RandomForestRegressor(n_estimators = 2000, max_depth = 20, n_jobs = -1, random_state = 42)

tFO = timer()
rf.fit(X_trainRF, y_train)
tSO = timer()


# In[45]:


print("Total fitting Random Forest time with optimized Scikit-learn: {} seconds".format(tSO - tFO))


# ### Default Scikit-learn

# In[46]:


from sklearnex import unpatch_sklearn
unpatch_sklearn()


# In[47]:


from sklearn.ensemble import RandomForestRegressor


# In[48]:


rf = RandomForestRegressor(n_estimators = 2000, max_depth = 20, n_jobs = -1, random_state = 42)

tFD = timer()
rf.fit(X_trainRF, y_train)
tSD = timer()


# In[49]:


print("Total fitting Random Forest time with default Scikit-learn: {} seconds".format(tSD - tFD))


# In[50]:


rf_speedup = round((tSD - tFD) / (tSO - tFO), 2)
HTML(f'<h2>RandomForest speedup: {rf_speedup}x</h2>'
     f'(from {round((tSD - tFD), 2)} to {round((tSO - tFO), 2)} seconds)')


# # Prediction

# In[51]:


submission_name =  'submit.csv'
submission['congestion'] = rf.predict(test_data)
submission.to_csv(submission_name, index = False)


# # Conclusion

# **Intel® Extension for Scikit-learn** gives you opportunities to:
# * Use your Scikit-learn code for training and inference without modification.
# * Get speed up your kernel

# *Please upvote if you liked it.*

# # Other notebooks with sklearnex usage

# ### [[predict sales] Stacking with scikit-learn-intelex](https://www.kaggle.com/alexeykolobyanin/predict-sales-stacking-with-scikit-learn-intelex)
# 
# ### [[TPS-Aug] NuSVR with Intel Extension for Sklearn](https://www.kaggle.com/alexeykolobyanin/tps-aug-nusvr-with-intel-extension-for-sklearn)
# 
# ### [Using scikit-learn-intelex for What's Cooking](https://www.kaggle.com/kppetrov/using-scikit-learn-intelex-for-what-s-cooking?scriptVersionId=58739642)
# 
# ### [Fast KNN using  scikit-learn-intelex for MNIST](https://www.kaggle.com/kppetrov/fast-knn-using-scikit-learn-intelex-for-mnist?scriptVersionId=58738635)
# 
# ### [Fast SVC using scikit-learn-intelex for MNIST](https://www.kaggle.com/kppetrov/fast-svc-using-scikit-learn-intelex-for-mnist?scriptVersionId=58739300)
# 
# ### [Fast SVC using scikit-learn-intelex for NLP](https://www.kaggle.com/kppetrov/fast-svc-using-scikit-learn-intelex-for-nlp?scriptVersionId=58739339)
# 
# ### [Fast AutoML with Intel Extension for Scikit-learn](https://www.kaggle.com/lordozvlad/fast-automl-with-intel-extension-for-scikit-learn)
# 
# ### [[Titanic] AutoML with Intel Extension for Sklearn](https://www.kaggle.com/lordozvlad/titanic-automl-with-intel-extension-for-sklearn)
