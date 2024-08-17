#!/usr/bin/env python
# coding: utf-8

# <a id="table"></a>
# <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Wine Quality Prediction</h1>
# 
# ![wine.jpg](attachment:1a43a364-3081-4b44-864b-d0822fd7b2f2.jpg)

# <a id="table"></a>
# <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Table of Contents</h1>
# 
# [1. Notebook Versions](#Notebook-Versions)
# 
# [2. Loading Libraries](#Loading-Libraries)
# 
# [3. Reading Data Files](#Reading-Data-Files)
# 
# [4. Data Exploration](#Data-Exploration)
# 
# [5. OneVsRest Overview](#OneVsRest-Overview)
# 
# [6 OneVsRest XGBoost](#OneVsRest:-XGBoost)
# 
# [7. XGBoost](#XGBoost)
# 
# [8. Feature Engineering](#Feature-Engineering)
# 
# [9. XGBoost Regressor](#XGBoost-Regressor)
# 
# [9. Model Comparison](#Model-Comparison)
# 
# 
# 
# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Notebook Versions</h1>
# 
# 1. Version 1 (01/30/2023)
#     * EDA 
#     * OneVsRest with XGBoost
# 
# 2. Version 2 (01/30/2023)
#     * Fixing bug
# 
# 3. Version 3 (01/30/2023)
#     * Adding section labels
# 
# 4. Version 4 (01/30/2023)
#     * Fixing bug
# 
# 5. Version 5 (02/01/2023)
#     * XGBoost modeling added
#     * Feature engineering added
# 
# 6. Version 6 (02/02/2023)
#     * Reducing the number of features in feature engineering
#     * Model comparison added
# 
# 7. Version 7 (02/02/2023)
#     * XGBoost hyperparamters updated
#     
# 8. Version 8 (02/04/2023)
#     * Cross-Validation code updated
#     * XGBoost Regressor added
#     
# 9. Version 9 (02/13/2023)
#     * Removing last section
#     
# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Loading Libraries</h1>    

# In[1]:


import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np

from tqdm import tqdm

from functools import partial
import scipy as sp

import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns
import plotly.express as px

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Reading Data Files</h1> 

# In[2]:


train = pd.read_csv('../input/playground-series-s3e5/train.csv')
test = pd.read_csv('../input/playground-series-s3e5/test.csv')
submission = pd.read_csv('../input/playground-series-s3e5/sample_submission.csv')

print('The dimession of the train dataset is:', train.shape)
print('The dimession of the test dataset is:', test.shape)


# In[3]:


train.info()


# In[4]:


test.info()


# In[5]:


train.head()


# In[6]:


train.describe()


# In[7]:


test.head()


# In[8]:


test.describe()


# > <div class="alert alert-block alert-info">
# <b>💡</b> There are no missing values neither in the train nor test datasets. Also, by a quick eye-ball comparison of the summary statistics of the train and test datasets, they seem to have similar distributions. 
# </div>

# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Data Exploration</h1> 

# In[9]:


round(100*train['quality'].value_counts() / train.shape[0], 2) 


# In[10]:


sns.countplot(x = 'quality', hue = 'quality', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above chart, we see that the data is imbalanced. Most of the data is related to quality lables 5, 6, and 7.</div>

# In[11]:


fig, axes = plt.subplots(1, 2, figsize = (22, 8))

sns.boxplot(ax = axes[0], x = 'quality', y = 'fixed acidity', hue = 'quality', data = train)
sns.boxplot(ax = axes[1], x = 'quality', y = 'volatile acidity', hue = 'quality', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the left panel, it seems that there is a very small increase in the fixed acidity medians (going from quality 4 to quality 7). On the right panel, there is decrease in the volatile acidity medians (going from quality 5 to quality 8).
# </div>

# In[12]:


fig, axes = plt.subplots(1, 2, figsize = (22, 8))

sns.boxplot(ax = axes[0], x = 'quality', y = 'citric acid', hue = 'quality', data = train)
sns.boxplot(ax = axes[1], x = 'quality', y = 'residual sugar', hue = 'quality', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the left panel, it seems that there is an increase in the citric acid medians (going from quality 4 to quality 8). On the right panel, there is no an abovious pattern when compare the residual sugar distribuition of the different wine qualities.</div>

# In[13]:


fig, axes = plt.subplots(1, 2, figsize = (22, 8))

sns.boxplot(ax = axes[0], x = 'quality', y = 'chlorides', hue = 'quality', data = train)
sns.boxplot(ax = axes[1], x = 'quality', y = 'free sulfur dioxide', hue = 'quality', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the left panel, there is no an abovious pattern when compare the residual sugar distribuitions of the different wine qualities. On the right panel, there is a decrease in the free sulfur dioxide emdian (going from quality 4 to quality 8).</div>

# In[14]:


fig, axes = plt.subplots(1, 2, figsize = (22, 8))

sns.boxplot(ax = axes[0], x = 'quality', y = 'total sulfur dioxide', hue = 'quality', data = train)
sns.boxplot(ax = axes[1], x = 'quality', y = 'density', hue = 'quality', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the left panel, there is an increase in the total sulfur dioxide medians (going from quality 3 to quality 5), then there is a decrease in the total sulfur dioxide medians (going from quality 5 to quality 8). On the right panel, there is a decrease in the density medians (going from quality 3 to quality 8).
# </div>

# In[15]:


fig, axes = plt.subplots(1, 2, figsize = (22, 8))

sns.boxplot(ax = axes[0], x = 'quality', y = 'pH', hue = 'quality', data = train)
sns.boxplot(ax = axes[1], x = 'quality', y = 'sulphates', hue = 'quality', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the left panel, there is no an abovious pattern when compare the pH distribuitions of the different wine qualities. On the right panel, there is an increase in the sulphates medians (going from quality 5 to quality 8).
# </div>

# In[16]:


plt.figure(figsize = (8.5, 7))
sns.boxplot(x = 'quality', y = 'alcohol', hue = 'quality', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> In the above chart, there is an abovious increase in the alcohol medians (going from quality 4 to quality 8).
# </div>
# 
# We next proceed to explore potential correlation among the predictor features in the train and test dataset.

# In[17]:


## Explore the correlation between all numerical features
corr_mat_train = train.drop(columns = ['Id', 'quality'], axis = 1).corr()
corr_mat_test = test.drop(columns = ['Id'], axis = 1).corr()

## Keep only correlation higher than a threshold
threshold = 0.3
corr_threshold_train = corr_mat_train[(corr_mat_train > threshold) | (corr_mat_train < -threshold)]
corr_threshold_test = corr_mat_test[(corr_mat_test > threshold) | (corr_mat_test < -threshold)]

fig, axes = plt.subplots(1, 2, figsize = (22, 8))
sns.heatmap(corr_threshold_train, annot = True, cmap = 'seismic', fmt = ".2f",
            linewidths = 0.5, cbar_kws={'shrink': .5},annot_kws={'size': 8}, ax = axes[0]).set_title('Correlations Among Features (in Train)')
sns.heatmap(corr_threshold_test, annot = True, cmap = 'seismic', fmt = ".2f",
            linewidths = 0.5, cbar_kws={'shrink': .5},annot_kws={'size': 8}, ax = axes[1]).set_title('Correlations Among Features (in Test)');


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above heatmaps, we see that the correlation among the features are the same in the train and test datasets.
# </div>
# 
# We next proceed to compare the distribution of the features in the train and test datasets.

# In[18]:


train_vis = train.drop(columns = 'quality', axis = 1).reset_index(drop = True).copy()
test_vis = test.copy()

train_vis['Dataset'] = 'Train'
test_vis['Dataset'] = 'Test'
data_tot = pd.concat([train_vis, test_vis], axis = 0).reset_index(drop = True)

fig, axes = plt.subplots(4, 3, figsize = (25, 20))

sns.kdeplot(ax = axes[0, 0], x = 'fixed acidity', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[0, 1], x = 'volatile acidity', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[0, 2], x = 'citric acid', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 0], x = 'residual sugar', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 1], x = 'chlorides', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 2], x = 'free sulfur dioxide', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[2, 0], x = 'total sulfur dioxide', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[2, 1], x = 'density', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[2, 2], x = 'pH', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[3, 0], x = 'sulphates', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[3, 1], x = 'alcohol', hue = 'Dataset', data = data_tot, fill = True);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above density plots, we see that the distributions of the features are very similar in the train and test datasets.
# </div>

# <div class="alert alert-block alert-info">
# <b>💡 Insights from EDA:</b><br> 
# <ul>
#     <li> Imbalanced dataset (not many 3, 4, and 8 quality labels). </li>
#     <li> There is an increase in the citric acid medians (going from quality 4 to quality 8). </li>
#     <li> There is a decrease in the free sulfur dioxide emdian (going from quality 4 to quality 8). </li>
#     <li> There is a decrease in the density medians (going from quality 3 to quality 8). </li>
#     <li> There is an increase in the sulphates medians (going from quality 5 to quality 8). </li>
#     <li> There is an abovious increase in the alcohol medians (going from quality 4 to quality 8).</li>
#     <li> The distributions of the features are very similar in the train and test datasets.</li>
#     <li> There is a moderate positive correlation between fixed acidity and citric acid.</li>
#     <li> There is a moderate negative correlation between fixed acidity and pH</li>
# </ul>
# </div>

# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">OneVsRest Overview</h1> 
# 
# In this section, we briefly describe the one-vs-rest (one-vs-all) stratefy for multi-class classification. In this approach, one classifier is fitted against one class. For each of the classifiers, the class is then fitted against all the other classes, producing a real-valued decision confidence score, instead of class labels. From the confidence score, the maximum value is picked up to get the final class label. The advantage of one-vs-all is its interpretability and efficiency.
# 
# ![chapter2_plot1.png](attachment:a587ce02-c33e-4fa8-81c7-04bb77e401a0.png)
# 
# Notice that this approach can be implemented with any of the standard classifiers such as logistic regression, XGBoost, etc. In the next section, we build a base-line model using the one-vs-rest strategy combined with XGBoost.

# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">OneVsRest: XGBoost</h1> 

# In[19]:


test_md = test.copy()

X = train.drop(columns = ['Id', 'quality'], axis = 1)
Y = train['quality']
test_md = test_md.drop(columns = 'Id', axis = 1)

XGB_cv_scores, XGB_imp = list(), list()
preds = list()

## Running 5 times CV
for i in range(5):
    
    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
                
        ## Building RF model
        XGB_md = XGBClassifier(tree_method = 'gpu_hist',
                               colsample_bytree = 0.7, 
                               gamma = 0.8, 
                               learning_rate = 0.01, 
                               max_depth = 5, 
                               min_child_weight = 10, 
                               n_estimators = 1000, 
                               subsample = 0.7)
        
        One_vs_Rest_XGB = OneVsRestClassifier(estimator = XGB_md).fit(X_train, Y_train)
        
        ## Predicting on X_test and test
        XGB_pred_1 = One_vs_Rest_XGB.predict(X_test)
        XGB_pred_2 = One_vs_Rest_XGB.predict(test_md)
        
        ## Computing roc-auc score
        XGB_cv_scores.append(cohen_kappa_score(Y_test, XGB_pred_1, weights = 'quadratic'))
        preds.append(XGB_pred_2)

XGB_cv_score = np.mean(XGB_cv_scores)    
print('The average oof quadratic weighted kapp score over 5-folds (run 5 times) is:', XGB_cv_score)


# In[20]:


XGB_preds_test = pd.DataFrame(preds).mode(axis = 0).loc[0, ]

submission['quality'] = XGB_preds_test.astype(int)
submission.head()


# In[21]:


submission.to_csv('XGB_baseline_onevsrest.csv', index = False)


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">XGBoost</h1> 
# 
# In this section, we model the data with XGBoost (wihtout using the OneVsRest strategy).

# In[22]:


test_md = test.copy()

X = train.drop(columns = ['Id', 'quality'], axis = 1)
Y = train['quality'] - 3
test_md = test_md.drop(columns = 'Id', axis = 1)

XGB_cv_scores, XGB_imp = list(), list()
preds = list()

## Running 5 times CV
for i in range(5):
    
    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
                
        ## Building RF model
        XGB_md = XGBClassifier(tree_method = 'gpu_hist',
                               colsample_bytree = 0.75, 
                               gamma = 0.8, 
                               learning_rate = 0.01, 
                               max_depth = 5, 
                               min_child_weight = 10, 
                               n_estimators = 1000, 
                               subsample = 0.7).fit(X_train, Y_train)
        XGB_imp.append(XGB_md.feature_importances_)
        
        ## Predicting on X_test and test
        XGB_pred_1 = XGB_md.predict(X_test)
        XGB_pred_2 = XGB_md.predict(test_md)
        
        ## Computing roc-auc score
        XGB_cv_scores.append(cohen_kappa_score(Y_test, XGB_pred_1, weights = 'quadratic'))
        preds.append(XGB_pred_2)

XGB_cv_score = np.mean(XGB_cv_scores)    
print('The average roc-auc score over 5-folds (run 5 times) is:', XGB_cv_score)


# In[23]:


plt.figure(figsize = (12, 14))
pd.DataFrame(XGB_imp, columns = X.columns).apply(np.mean, axis = 0).sort_values().plot(kind = 'barh');
plt.xlabel('XGBoost Score')
plt.ylabel('Feature')
plt.show();


# From the above chart, we see that, based on the XGBoost importance score, alcohol and sulphates are the top two most import features to predict wine quality.

# In[24]:


XGB_preds_test = pd.DataFrame(preds).mode(axis = 0).loc[0, ] + 3

submission['quality'] = XGB_preds_test.astype(int)
submission.head()


# In[25]:


submission.to_csv('XGB_baseline_raw.csv', index = False)


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Feature Engineering</h1> 
# 
# In this section, I implemented a couple of features suggest in this [post](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/382698).

# In[26]:


train['alcohol_density'] = train['alcohol']  * train['density']
train['sulphate/density'] = train['sulphates']  / train['density']

test['alcohol_density'] = test['alcohol']  * test['density']
test['sulphate/density'] = test['sulphates']  / test['density']

test_md = test.copy()

X = train.drop(columns = ['Id', 'quality'], axis = 1)
Y = train['quality'] - 3
test_md = test_md.drop(columns = 'Id', axis = 1)

XGB_cv_scores, XGB_imp = list(), list()
preds = list()

## Running 5 times CV
for i in range(5):
    
    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
                
        ## Building RF model
        XGB_md = XGBClassifier(tree_method = 'hist',
                               colsample_bytree = 0.75, 
                               gamma = 0.8, 
                               learning_rate = 0.01, 
                               max_depth = 5, 
                               min_child_weight = 10, 
                               n_estimators = 1000, 
                               subsample = 0.7).fit(X_train, Y_train)
        XGB_imp.append(XGB_md.feature_importances_)
        
        ## Predicting on X_test and test
        XGB_pred_1 = XGB_md.predict(X_test)
        XGB_pred_2 = XGB_md.predict(test_md)
        
        ## Computing roc-auc score
        XGB_cv_scores.append(cohen_kappa_score(Y_test, XGB_pred_1, weights = 'quadratic'))
        preds.append(XGB_pred_2)

XGB_cv_score = np.mean(XGB_cv_scores)    
print('The average roc-auc score over 5-folds (run 5 times) is:', XGB_cv_score)


# In[27]:


plt.figure(figsize = (12, 14))
pd.DataFrame(XGB_imp, columns = X.columns).apply(np.mean, axis = 0).sort_values().plot(kind = 'barh');
plt.xlabel('XGBoost Score')
plt.ylabel('Feature')
plt.show();


# From the above chart, we see that the both the engineered featues are ranked among the top three most important features in the XGBoost model.

# In[28]:


XGB_preds_test = pd.DataFrame(preds).mode(axis = 0).loc[0, ] + 3

submission['quality'] = XGB_preds_test.astype(int)
submission.head()


# In[29]:


submission.to_csv('XGB_baseline_raw_FE.csv', index = False)


# Next, we proceed to build a XGBoost model only using the top four features from the previous model result. My hunch is that we may be able to reach the same model performance with less number of input features. 

# In[30]:


test_md = test.copy()

X = train[['sulphate/density', 'alcohol_density', 'alcohol', 'sulphates']]
Y = train['quality'] - 3

test_md = test_md[['sulphate/density', 'alcohol_density', 'alcohol', 'sulphates']]

XGB_cv_scores = list()
preds = list()

## Running 5 times CV
for i in range(5):
    
    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
                
        ## Building RF model
        XGB_md = XGBClassifier(tree_method = 'hist',
                               colsample_bytree = 0.75, 
                               gamma = 0.8, 
                               learning_rate = 0.01, 
                               max_depth = 5, 
                               min_child_weight = 10, 
                               n_estimators = 1000, 
                               subsample = 0.7).fit(X_train, Y_train)
        
        ## Predicting on X_test and test
        XGB_pred_1 = XGB_md.predict(X_test)
        XGB_pred_2 = XGB_md.predict(test_md)
        
        ## Computing roc-auc score
        XGB_cv_scores.append(cohen_kappa_score(Y_test, XGB_pred_1, weights = 'quadratic'))
        preds.append(XGB_pred_2)

XGB_cv_score = np.mean(XGB_cv_scores)    
print('The average roc-auc score over 5-folds (run 5 times) is:', XGB_cv_score)


# From the above, we see that only using four features, the CV-score increased significantly.

# In[31]:


XGB_preds_test = pd.DataFrame(preds).mode(axis = 0).loc[0, ] + 3

submission['quality'] = XGB_preds_test.astype(int)
submission.head()


# In[32]:


submission.to_csv('XGB_FE_top_4.csv', index = False)


# After tuning the hyper-parameters with the optuna framework, we re-run the XGBoost model.

# In[33]:


XGB_cv_scores, XGB_imp = list(), list()
preds = list()

skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    
for train_ix, test_ix in skf.split(X, Y):
        
    ## Splitting the data 
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
                
    ## Building RF model
    XGB_md = XGBClassifier(tree_method = 'hist',
                           colsample_bytree = 0.7, 
                           gamma = 5.5, 
                           learning_rate = 0.031, 
                           max_depth = 5, 
                           min_child_weight = 68, 
                           n_estimators = 8800, 
                           subsample = 0.41, 
                           random_state = 42).fit(X_train, Y_train)
    XGB_imp.append(XGB_md.feature_importances_)
        
    ## Predicting on X_test and test
    XGB_pred_1 = XGB_md.predict(X_test)
    XGB_pred_2 = XGB_md.predict(test_md)
        
    ## Computing roc-auc score
    XGB_cv_scores.append(cohen_kappa_score(Y_test, XGB_pred_1, weights = 'quadratic'))
    preds.append(XGB_pred_2)

XGB_cv_score = np.mean(XGB_cv_scores)    
print('The average roc-auc score over 5-folds (run 5 times) is:', XGB_cv_score)


# In[34]:


XGB_preds_test = pd.DataFrame(preds).mode(axis = 0).loc[0, ] + 3

submission['quality'] = XGB_preds_test.astype(int)
submission.head()


# In[35]:


submission.to_csv('XGB_FE_top_4_tuned.csv', index = False)


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">XGBoost Regressor</h1>
# 
# Based on few discussions from the discussion forum, I decided to approach this task as regression instead of classification. For references see either this [post](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/382525) or this [post](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/382525). Something important to keep in mind is how the regression predictions are going to be transformed into labels. I used the [OptimizedRounder](https://www.kaggle.com/competitions/petfinder-adoption-prediction/discussion/76107) approach proposed by abhisek.

# In[36]:


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 3
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 4
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 5
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 6
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 7
            else:
                X_p[i] = 8

        ll = cohen_kappa_score(y, X_p, weights = 'quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = [3.5, 4.5, 5.5, 6.5, 7.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method = 'nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 3
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 4
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 5
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 6
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 7
            else:
                X_p[i] = 8
        return X_p

    def coefficients(self):
        return self.coef_['x']


# After defining the `OptimizedRounder`, we proceed to evaluate the performance of the model. Notice that the selection of hyper-parameters was conducted using the optuna framework.

# In[37]:


test_md = test.copy()

X = train[['sulphate/density', 'alcohol_density', 'alcohol', 'sulphates']]
Y = train['quality'] 

test_md = test_md[['sulphate/density', 'alcohol_density', 'alcohol', 'sulphates']]

XGB_cv_scores, XGB_imp = list(), list()
preds = list()

skf = KFold(n_splits = 5, random_state = 42, shuffle = True)
    
for train_ix, test_ix in skf.split(X, Y):
        
    ## Splitting the data 
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
                
    ## Building RF model
    XGB_md = XGBRegressor(tree_method = 'hist',
                              colsample_bytree = 0.2874, 
                              gamma = 8, 
                              learning_rate = 0.05592, 
                              max_depth = 6, 
                              min_child_weight = 30, 
                              n_estimators = 5235, 
                              subsample = 0.7574, 
                              random_state = 42).fit(X_train, Y_train)
    XGB_imp.append(XGB_md.feature_importances_)
        
    ## Predicting on X_test and test
    XGB_pred_1 = XGB_md.predict(X_test)
    XGB_pred_2 = XGB_md.predict(test_md)
        
    ## Applying Optimal Rounder (using abhishek approach)
    optR = OptimizedRounder()
    optR.fit(XGB_pred_1, Y_test)
    coef = optR.coefficients()
    XGB_pred_1 = optR.predict(XGB_pred_1, coef).astype(int)
    XGB_pred_2 = optR.predict(XGB_pred_2, coef).astype(int)
        
    ## Computing roc-auc score
    XGB_cv_scores.append(cohen_kappa_score(Y_test, XGB_pred_1, weights = 'quadratic'))
    preds.append(XGB_pred_2)

XGB_cv_score = np.mean(XGB_cv_scores)    
print('The average quadratic weighted kappa score over 5-folds is:', XGB_cv_score)


# In[38]:


XGB_preds_test = pd.DataFrame(preds).mode(axis = 0).loc[0, ]

submission['quality'] = XGB_preds_test.astype(int)
submission.head()


# In[39]:


submission.to_csv('XGB_Reg_FE_top_4_tuned.csv', index = False)


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Model Comparison</h1>
# 
# | Model | Features | CV | LB |
# | --- | --- | --- | --- |
# | OneVsRest with XGBoost | all raw features | 0.49699 | 0.51894 |
# | XGBoost | all raw featueres + sulphate/density | 0.49801 | 0.53390 |
# | XGBoost | all raw featueres + sulphate/density + alcohol_density | 0.50225 | 0.54121 |
# | XGBoost | sulphate/density + alcohol_density + alcohol + sulphates | 0.53176 | 0.54429 |
# | XGBoost (tuned) | sulphate/density + alcohol_density + alcohol + sulphates | 0.52807 | 0.57265 |
# | XGBoost Regressor (tuned) | sulphate/density + alcohol_density + alcohol + sulphates | 0.57196 | 0.58267 |
