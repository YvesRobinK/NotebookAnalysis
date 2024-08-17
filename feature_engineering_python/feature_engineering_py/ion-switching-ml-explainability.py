#!/usr/bin/env python
# coding: utf-8

# ## <font size='5' color='blue'>Introduction</font>
# 
# ![](https://blogs.sas.com/content/subconsciousmusings/files/2018/06/blackboxmodels.png)
# 
# 
# Through this notebook I want to explore the and extract insights from models and try to answer some of the below questions
# 1. Which are the common approches and tools used for Interpreting ML models?
# 2. Which are the most important features that affect our prediction?
# 3. How these features affect the target?
# 4. What insights can we extract  from models?
# 5. How to interpret predictions of single sample?

# <font size='5' color='red'>If you like this notebook,please consider leaving an upvote ⬆️</font>

# ## <font size='3' color='blue'> Loading required packages</font>

# In[1]:


seed_random = 42
window_sizes = [10, 50]


# In[2]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, make_scorer
import lightgbm as lgb
import xgboost as xgb
from pykalman import KalmanFilter
from functools import partial
import scipy as sp
import time
import datetime
import gc
from sklearn.tree import DecisionTreeClassifier
import shap


# In[3]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        if col != 'time':
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# ## <font size='4' color='blue'>Feature Engineering</font>
# Below I have  the most common features used in some of the kernels by excellent kagglers.I really appreciate their efforts and thank them for making it public.

# In[4]:


train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')


# ### FE-2 - thanks to the kernels:
# * https://www.kaggle.com/teejmahal20/regression-with-optimized-rounder
# * https://www.kaggle.com/pestipeti/eda-ion-switching

# In[5]:


get_ipython().run_cell_magic('time', '', 'for window in window_sizes:\n    train["rolling_mean_" + str(window)] = train[\'signal\'].rolling(window=window).mean()\n    train["rolling_std_" + str(window)] = train[\'signal\'].rolling(window=window).std()\n    train["rolling_var_" + str(window)] = train[\'signal\'].rolling(window=window).var()\n    train["rolling_min_" + str(window)] = train[\'signal\'].rolling(window=window).min()\n    train["rolling_max_" + str(window)] = train[\'signal\'].rolling(window=window).max()\n    \n    #train["rolling_min_max_ratio_" + str(window)] = train["rolling_min_" + str(window)] / train["rolling_max_" + str(window)]\n    #train["rolling_min_max_diff_" + str(window)] = train["rolling_max_" + str(window)] - train["rolling_min_" + str(window)]\n    \n    a = (train[\'signal\'] - train[\'rolling_min_\' + str(window)]) / (train[\'rolling_max_\' + str(window)] - train[\'rolling_min_\' + str(window)])\n    train["norm_" + str(window)] = a * (np.floor(train[\'rolling_max_\' + str(window)]) - np.ceil(train[\'rolling_min_\' + str(window)]))\n    \ntrain = train.replace([np.inf, -np.inf], np.nan)    \ntrain.fillna(0, inplace=True)\n')


# In[6]:


get_ipython().run_cell_magic('time', '', 'for window in window_sizes:\n    test["rolling_mean_" + str(window)] = test[\'signal\'].rolling(window=window).mean()\n    test["rolling_std_" + str(window)] = test[\'signal\'].rolling(window=window).std()\n    test["rolling_var_" + str(window)] = test[\'signal\'].rolling(window=window).var()\n    test["rolling_min_" + str(window)] = test[\'signal\'].rolling(window=window).min()\n    test["rolling_max_" + str(window)] = test[\'signal\'].rolling(window=window).max()\n    \n    #test["rolling_min_max_ratio_" + str(window)] = test["rolling_min_" + str(window)] / test["rolling_max_" + str(window)]\n    #test["rolling_min_max_diff_" + str(window)] = test["rolling_max_" + str(window)] - test["rolling_min_" + str(window)]\n\n    \n    a = (test[\'signal\'] - test[\'rolling_min_\' + str(window)]) / (test[\'rolling_max_\' + str(window)] - test[\'rolling_min_\' + str(window)])\n    test["norm_" + str(window)] = a * (np.floor(test[\'rolling_max_\' + str(window)]) - np.ceil(test[\'rolling_min_\' + str(window)]))\n\ntest = test.replace([np.inf, -np.inf], np.nan)    \ntest.fillna(0, inplace=True)\n')


# ### FE-3 - thanks to 
# * https://www.kaggle.com/jazivxt/physically-possible
# * https://www.kaggle.com/siavrez/simple-eda-model

# In[7]:


get_ipython().run_cell_magic('time', '', "def features(df):\n    df = df.sort_values(by=['time']).reset_index(drop=True)\n    df.index = ((df.time * 10_000) - 1).values\n    df['batch'] = df.index // 25_000\n    df['batch_index'] = df.index  - (df.batch * 25_000)\n    df['batch_slices'] = df['batch_index']  // 2500\n    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)\n    \n    for c in ['batch','batch_slices2']:\n        d = {}\n        d['mean'+c] = df.groupby([c])['signal'].mean()\n        d['median'+c] = df.groupby([c])['signal'].median()\n        d['max'+c] = df.groupby([c])['signal'].max()\n        d['min'+c] = df.groupby([c])['signal'].min()\n        d['std'+c] = df.groupby([c])['signal'].std()\n        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))\n        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))\n        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))\n        d['range'+c] = d['max'+c] - d['min'+c]\n        d['maxtomin'+c] = d['max'+c] / d['min'+c]\n        d['abs_avg'+c] = (d['abs_min'+c] + d['abs_max'+c]) / 2\n        for v in d:\n            df[v] = df[c].map(d[v].to_dict())\n\n    \n    # add shifts_1\n    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])\n    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]\n    for i in df[df['batch_index']==0].index:\n        df['signal_shift_+1'][i] = np.nan\n    for i in df[df['batch_index']==49999].index:\n        df['signal_shift_-1'][i] = np.nan\n    \n    # add shifts_2 - my upgrade\n    df['signal_shift_+2'] = [0,] + [1,] + list(df['signal'].values[:-2])\n    df['signal_shift_-2'] = list(df['signal'].values[2:]) + [0] + [1]\n    for i in df[df['batch_index']==0].index:\n        df['signal_shift_+2'][i] = np.nan\n    for i in df[df['batch_index']==1].index:\n        df['signal_shift_+2'][i] = np.nan\n    for i in df[df['batch_index']==49999].index:\n        df['signal_shift_-2'][i] = np.nan\n    for i in df[df['batch_index']==49998].index:\n        df['signal_shift_-2'][i] = np.nan\n    \n    df = df.drop(columns=['batch', 'batch_index', 'batch_slices', 'batch_slices2'])\n\n    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels']]:\n        df[c+'_msignal'] = df[c] - df['signal']\n        \n    df = df.replace([np.inf, -np.inf], np.nan)    \n    df.fillna(0, inplace=True)\n    gc.collect()\n    return df\n\ntrain = features(train)\ntest = features(test)\n")


# In[8]:


train = reduce_mem_usage(train)


# In[9]:


test = reduce_mem_usage(test)


# In[10]:


y = train['open_channels']
col = [c for c in train.columns if c not in ['time', 'open_channels', 'group', 'medianbatch', 'abs_avgbatch', 'abs_maxbatch']]


# In[11]:


train.head()


# ## <font size='4' color='blue'> Build Model</font>
# 
# Remember building a fine model is not our focus in this notebook,so we will use the default settings and make a LGB Classifier model.
# 

# In[12]:


# Thanks to https://www.kaggle.com/siavrez/simple-eda-model
def MacroF1Metric(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(int)
    score = f1_score(labels, preds, average = 'macro')
    return ('MacroF1Metric', score, True)


# In[13]:


get_ipython().run_cell_magic('time', '', '# Thanks to https://www.kaggle.com/jazivxt/physically-possible with tuning from https://www.kaggle.com/siavrez/simple-eda-model and my tuning\nX_train, X_valid, y_train, y_valid = train_test_split(train[col], y, test_size=0.01, random_state=seed_random)\n\nmodel=lgb.LGBMClassifier(n_estimators=10)\nmodel.fit(X_train,y_train)\n')


# ## <font size='4' color='green'> Feature importances</font>

# Before we get into using external packages for interpreting our model,we can first take a look into `feature importances`.
# But we always do a mistake using default feature_importance function.We only consider the default importace_type and judge the model.There are 2 importance_type for lgb tree models.
# - `split`: Number of times a feature was used to split nodes in the tree.
# - `gain` : The information gain from the feature.
# 
# first we will check split importance_type

# In[14]:


fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
lgb.plot_importance(model,ax = axes,height = 0.5,importance_type='split')
plt.show();plt.close()
gc.collect()


# We can see that the feature `minibatch_msignal` is the most used feature inorder to split the nodes in the tree.We will do further inspection about this feature later.Next we will plot the `gain` from each feature.

# In[15]:


fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
lgb.plot_importance(model,ax = axes,height = 0.5,importance_type='gain')
plt.show();plt.close()
gc.collect()


# ##  <font size='4' color='red'> Permutation Importance</font>
# 
# In this section we will answer following question:
# 
# 1. What features have the biggest impact on predictions?
# 2. how to extract insights from models?
# 

# In[16]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=1).fit(X_valid, y_valid)



# In[17]:


eli5.show_weights(perm, feature_names = X_valid.columns.tolist(), top=150)


#  What can be inferred from the above?
# 
#  1. As you move down the top of the graph, the importance of the feature decreases.
#  2. The features that are shown in green indicate that they have a positive impact on our prediction
#  3. The features that are shown in white indicate that they have no effect on our prediction
#  4. The features shown in red indicate that they have a negative impact on our prediction
#  5. The most important feature was `minibatch_msignal`.
# 

# ## <font size='4' color='brown'>Partial dependancy plots.</font>
# 
# 
# A partial dependence (PD) plot depicts the functional relationship between a small number of input variables and predictions. They show how the predictions partially depend on values of the input variables of interest.

# In[18]:


features=X_valid.columns.tolist()
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(X_train, y_train)


# In[19]:


from sklearn import tree
import graphviz
tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=features)


# In[20]:


graphviz.Source(tree_graph)


# 
# 
# As guidance to read the tree:
# 
#    - Leaves with children show their splitting criterion on the top
#    - The pair of values at the bottom show the count of False values and True values for the target respectively, of data points in that node of the tree.
# 
# Next,we will plot the partial dependency plot.
# Using PD we can answer **how a selected variable affects the prediction?**

# In[21]:


from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=X_valid.iloc[:10000], model_features=features, feature='minbatch_msignal')

# plot it
pdp.pdp_plot(pdp_goals, 'minbatch_msignal')
plt.show()


# 
# 
# A few items are worth pointing out as you interpret this plot
# 
# - The y axis is interpreted as change in the prediction from what it would be predicted at the baseline or leftmost value.
# - A blue shaded area indicates level of confidence
#  
#  
# From this particular graph,the first image shows that the probability of predicting the sample to class 0 increasing as the `minbatch_msginal` increases from -4 to 0.

# Let's check the PD plot of `stdbatch` feature

# In[22]:


pdp.pdp_plot(pdp_goals, 'stdbatch')
plt.show()


# ##  <font size='4' color='blue'>Two variable interaction plot</font>
# 
# PD plots look at the variable of interest across a specified range. At each value of the variable, the model is evaluated for all observations of the other model inputs, and the output is then averaged. Thus, the relationship they depict is only valid if the variable of interest does not interact strongly with other model inputs.
# 
# Since variable interactions are common in actual practice, you can use higher-order (such as two-way) partial dependence plots to check for interactions among specific model variables.
# 
# Now we will check the interaction between the variables `miibatch_msignal` and `stdbatch`.
# 
# Using intercation plot can help us find our **how interaction between these variables affects the prediction?**

# In[23]:


features_to_plot = ['minbatch_msignal', 'stdbatch']
inter1  =  pdp.pdp_interact(model=tree_model, dataset=X_valid.iloc[:10000], model_features=features, features=features_to_plot,)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour',which_classes=[0,1])
plt.show()


# Here you can see how these variables interact and affects the prediction.
# 
# 
# 
# 
# ## <font size='4' color='green'>SHAP Values</font>
# 
# SHAP (SHapley Additive exPlanations) is a unified approach to explain the output of any machine learning model. SHAP connects game theory with local explanations, uniting several previous methods and representing the only possible consistent and locally accurate additive feature attribution method based on expectations (see the SHAP NIPS paper for details).
# 
# ![](https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_diagram.png)
# 
# By using shapely we can understand **how model works for individual predictions?**

# ## SHAP Summary Plot
# The summary plot combines feature importance with feature effects. Each point on the summary plot is a Shapley value for a feature and an instance. The position on the y-axis is determined by the feature and on the x-axis by the Shapley value. The color represents the value of the feature from low to high. Overlapping points are jittered in y-axis direction, so we get a sense of the distribution of the Shapley values per feature. The features are ordered according to their importance

# In[24]:


explainer = shap.TreeExplainer(model)
shap_values=explainer.shap_values(X_valid)


# In[25]:


shap.summary_plot(shap_values, X_valid)


# Here you can see that how each feature has affected the prediction of each class.

# ## Individual SHAP Value Plot — Local Interpretability
#  You can visualize feature attributions such as Shapley values as “forces”. Each feature value is a force that either increases or decreases the prediction. The prediction starts from the baseline. The baseline for Shapley values is the average of all predictions. In the plot, each Shapley value is an arrow that pushes to increase (positive value) or decrease (negative value) the prediction. These forces balance each other out at the actual prediction of the data instance.

# In[26]:


row_to_show = 1
data_for_prediction = X_valid.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)




# In[27]:


shap_values = explainer.shap_values(data_for_prediction_array)


# For class 0

# In[28]:


shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[1], data_for_prediction)


# For class 1

# In[29]:


shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)


# Here the features on the left (red) moves the prediction away from the average and the features on the right (blue) moves the prediction towards the average.

# ## <font size='3' color='red'>SHAP dependence plot</font>

# SHAP dependence plots show the effect of a single feature across the whole dataset. They plot a feature's value vs. the SHAP value of that feature across many samples. SHAP dependence plots are similar to partial dependence plots, but account for the interaction effects present in the features, and are only defined in regions of the input space supported by data. The vertical dispersion of SHAP values at a single feature value is driven by interaction effects, and another feature is chosen for coloring to highlight possible interaction

# We will plot this for some of the important features

# In[30]:


shap_values=explainer.shap_values(X_valid)
for name in ['minbatch_msignal',"stdbatch","rangebatch"]:
    shap.dependence_plot(name,shap_values[1],X_valid)


# In[ ]:





# <font size='5' color='red'>If you like this notebook,please consider leaving an upvote ⬆️ <font size='3' color='blue'>Thank you</font> 

# In[ ]:




