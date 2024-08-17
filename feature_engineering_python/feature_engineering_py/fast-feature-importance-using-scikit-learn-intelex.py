#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:rgba(0, 167, 255, 0.6);border-radius:5px;display:fill">
#     <h1><center>Tabular Playground Series - Nov 2021
# </div>

# <center><a><img src="https://i.ibb.co/PWvpT9F/header.png" alt="header" border="0" width=800 height=400 class="center"></a>

# For classical machine learning algorithms, we often use the most popular Python library, Scikit-learn. With Scikit-learn you can fit models and search for optimal parameters, but it sometimes works for hours. Speeding up this process is something anyone who uses Scikit-learn would be interested in.
# 
# I want to show you how to use Scikit-learn library and get the results faster without changing the code. To do this, we will make use of another Python library, [**Intel® Extension for Scikit-learn***](https://github.com/intel/scikit-learn-intelex). It accelerates Scikit-learn and does not require you to change the code written for Scikit-learn.
# 
# I will show you how to **speed up** your kernel without changing your code!

# <div style="background-color:rgba(0, 167, 255, 0.6);border-radius:5px;display:fill">
#     <h1><center>Importing Libraries and Data</center></h1>
# </div>

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt


# ### Reading Data

# In[2]:


PATH_TRAIN      = '../input/tabular-playground-series-nov-2021/train.csv'
PATH_TEST       = '../input/tabular-playground-series-nov-2021/test.csv'
PATH_SUBMISSION = '../input/tabular-playground-series-nov-2021/sample_submission.csv'


# In[3]:


id_column  = 'id'
train_data = pd.read_csv(PATH_TRAIN, index_col = id_column)
test_data  = pd.read_csv(PATH_TEST, index_col = id_column)
submission = pd.read_csv(PATH_SUBMISSION, index_col = id_column)


# ### Reduce DataFrame memory usage
# 
# Since data is quite big for Kaggle notebook instance RAM, we need to reduce memory usage by switching data types.

# In[4]:


label = 'target'
features = [col for col in train_data.columns if 'f' in col]

cont_features = []
disc_features = []

for col in features:
    if train_data[col].dtype=='float64':
        cont_features.append(col)
    else:
        disc_features.append(col)

train_data[cont_features] = train_data[cont_features].astype('float32')
train_data[disc_features] = train_data[disc_features].astype('uint8')
train_data[cont_features] = train_data[cont_features].astype('float32')
train_data[disc_features] = train_data[disc_features].astype('uint8')


# In[5]:


train_data.info()


# Memory usage was reduced from 467 MB to 238 MB

# Collect garbage to reduce memory usage

# In[6]:


import gc

gc.collect()


# ### Intel® Extension for Scikit-learn installation:

# In[7]:


get_ipython().system('pip install scikit-learn-intelex -q --progress-bar off > /dev/null 2>&1')


# ### Accelerate Scikit-learn with two lines of code:

# In[8]:


from sklearnex import patch_sklearn
patch_sklearn()


# Setup logging to track accelerated cases:

# In[9]:


import logging

logger = logging.getLogger()
fh     = logging.FileHandler('log.txt')

fh.setLevel(10)
logger.addHandler(fh)


# <div style="background-color:rgba(0, 167, 255, 0.6);border-radius:5px;display:fill">
#     <h1><center>Feature importance</center></h1>
# </div>

# One of the most basic questions we might ask of a model is: What features have the biggest impact on predictions?
# 
# This concept is called feature importance.
# 
# There are multiple ways to measure feature importance. In this kernel we compare two way: default Scikit-learn permutation importance and feature importance using library ELI5.

# Let's start with default Scikit-learn permutation importance.

# In[10]:


X, y = train_data.drop(['target'], axis = 1), train_data['target']


# In[11]:


from sklearn.model_selection import train_test_split
from timeit import default_timer as timer

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[12]:


from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

timePerF = timer()
modelRF  = RandomForestClassifier(random_state = 42).fit(X_train, y_train)
per      = permutation_importance(modelRF, X_val, y_val, random_state = 42)
timePerS = timer()


# In[13]:


print("Total time with Intel Extension: {} seconds".format(timePerS - timePerF))


# In[14]:


newPer = [(per.importances_mean[index], per.importances_std[index], index) for index in range(len(per.importances_mean)) if per.importances_mean[index] >= 0.001]
newPer.sort()
newPer = newPer[::-1]


# In[15]:


from plotly import figure_factory as FF

table_data = [['Weight', 'Feature']]

for (mean, std, index) in newPer[:10]:
    temp = [str(round(mean, 5)) + " ± " + str(round(std, 5)), str(index)]
    table_data.append(temp)
    
figure = FF.create_table(table_data, height_constant=20)
figure.layout.width = 250


# In[16]:


figure.show()


# In[17]:


rf_features = []
for (mean, std, index) in newPer:
    rf_features.append(index)


# In[18]:


rf_features[:5]


# ### ELI5

# ELI5 provides a way to compute feature importances for any black-box estimator by measuring how score decreases when a feature is not available.

# In[19]:


import eli5
from eli5.sklearn import PermutationImportance
from timeit import default_timer as timer


# In[20]:


timeFirstI  = timer()
modelRF     = RandomForestClassifier(random_state = 42).fit(X_train, y_train)
perm        = PermutationImportance(modelRF, random_state = 42).fit(X_val, y_val)
timeSecondI = timer()


# In[21]:


print("Total time with Intel Extension: {} seconds".format(timeSecondI - timeFirstI))


# In[22]:


eli5.show_weights(perm, feature_names = X.columns.tolist())


# In[23]:


pi_features = eli5.explain_weights_df(perm, feature_names = X_train.columns.tolist())
pi_features = pi_features.loc[pi_features['weight'] >= 0.001]['feature'].tolist()


# In[24]:


pi_features[:5]


# In[25]:


X_trainRF = X_train.loc[:, rf_features]
X_trainPI = X_train.loc[:, pi_features]


# In[26]:


X_trainRF[:5]


# In[27]:


X_trainPI[:5]


# In[28]:


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

kf = KFold(n_splits = 5)

res = []
for random_state in [42, 55, 777]:
    slf = RandomForestClassifier(random_state = random_state)
    for train_index, test_index in kf.split(X_trainRF):
        X_trains, X_tests = X_trainRF.iloc[train_index], X_trainRF.iloc[test_index]
        y_trains, y_tests = y_train.iloc[train_index], y_train.iloc[test_index]
        slf.fit(X_trains, y_trains)
        res.append(roc_auc_score(y_tests, slf.predict_proba(X_tests)[:, 1]))


# In[29]:


print("Roc AUC score on default Scikit-learn", round(np.average(res), 3))


# In[30]:


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

kf = KFold(n_splits = 5)

res = []
for random_state in [42, 55, 777]:
    slf = RandomForestClassifier(random_state = random_state)
    for train_index, test_index in kf.split(X_trainPI):
        X_trains, X_tests = X_trainPI.iloc[train_index], X_trainPI.iloc[test_index]
        y_trains, y_tests = y_train.iloc[train_index], y_train.iloc[test_index]
        slf.fit(X_trains, y_trains)
        res.append(roc_auc_score(y_tests, slf.predict_proba(X_tests)[:, 1]))


# In[31]:


print("Roc AUC score using ELI5", round(np.average(res), 3))


# ### Accelerated functions:

# In[32]:


get_ipython().system("cat log.txt | grep 'running accelerated version' | sort | uniq")


# ### Default Scikit-learn

# In[33]:


from sklearnex import unpatch_sklearn
unpatch_sklearn()


# In[34]:


from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

timePerDF = timer()
modelRF   = RandomForestClassifier(random_state = 42).fit(X_train, y_train)
per       = permutation_importance(modelRF, X_val, y_val, random_state = 42)
timePerDS = timer()


# In[35]:


print("Total time with default Scikit-learn: {} seconds".format(timePerDS - timePerDF))


# In[36]:


import eli5
from eli5.sklearn import PermutationImportance
from timeit import default_timer as timer
from sklearn.ensemble import RandomForestClassifier


# In[37]:


timeFirstD  = timer()
modelRF     = RandomForestClassifier(random_state = 42).fit(X_train, y_train)
perm        = PermutationImportance(modelRF, random_state = 42).fit(X_val, y_val)
timeSecondD = timer()


# In[38]:


print("Total time with default Scikit-learn: {} seconds".format(timeSecondD - timeFirstD))


# In[39]:


eli5.show_weights(perm, feature_names = X.columns.tolist())


# In[40]:


eli5_speedup = round((timeSecondD - timeFirstD) / (timeSecondI - timeFirstI), 2)
pi_speedup = round((timePerDS - timePerDF) / (timePerS - timePerF), 2)
HTML(f'<h2>ELI5 speedup: {eli5_speedup}x</h2>'
     f'(from {round((timeSecondD - timeFirstD), 2)} to {round((timeSecondI - timeFirstI), 2)} seconds)'
    f'<h2>Scikit-learn permutation importance speedup: {pi_speedup}x</h2>'
     f'(from {round((timePerDS - timePerDF), 2)} to {round((timePerS - timePerF), 2)} seconds)')


# <div style="background-color:rgba(0, 167, 255, 0.6);border-radius:5px;display:fill">
#     <h1><center>Conclusion</center></h1>
# </div>

# **Intel® Extension for Scikit-learn** gives you opportunities to:
# * Use your Scikit-learn code for training and inference without modification.
# * Get speed up your kernel

# *Please upvote if you liked it.*

# <div style="background-color:rgba(0, 167, 255, 0.6);border-radius:5px;display:fill">
#     <h1><center>Other notebooks with sklearnex usage</center></h1>
# </div>

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
