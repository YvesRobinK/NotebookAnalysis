#!/usr/bin/env python
# coding: utf-8

# ![Credit_Card_Fraud_Detection.png](attachment:6be4c868-0478-47d5-92c1-6a48151ac5c3.png)

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
# [5. XGBoost Modeling](#XGBoost-Modeling)
# 
# [6. LightGBM Modeling](#LightGBM-Modeling)
# 
# [7. Model Performance Comparisson](#Model-Performance-Comparisson)
# 
# [8. Feature Engineering](#Feature-Engineering)
# 
# 
# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Notebook Versions</h1>
# 
# 1. Version 1 (01/23/2023)
#     * EDA 
#     * XGBoost Modeling
#     
# 2. Version 2 (01/23/2023)
#     * Fixing bug in submission
#     
# 3. Version 3 (01/24/2023)
#     * LightGBM Modeling added
#     * Modeling Comparisson added
#     
# 4. Version 4 (01/24/2023)
#     * Fixed bug
#     
# 5. Version 5 (01/24/2023)
#     * Saving predictions
#     
# 6. Version 6 (01/25/2023)
#     * Feature engineering added (time feature)
# 
# 7. Version 7 (01/30/2023)
#     * Include perfect cross valdiation (https://www.kaggle.com/competitions/playground-series-s3e4/discussion/381415)
#     
# 8. Version 8 (01/30/2023)
#     * Fixing bud in drop statement.
# 
# 9. Version 9 (02/15/2023)
#     * Linking to GitHub
#     
# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Loading Libraries</h1>    

# In[1]:


import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns
import plotly.express as px

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Reading Data Files</h1> 

# In[2]:


train = pd.read_csv('../input/playground-series-s3e4/train.csv')
test = pd.read_csv('../input/playground-series-s3e4/test.csv')
submission = pd.read_csv('../input/playground-series-s3e4/sample_submission.csv')

print('The dimession of the train dataset is:', train.shape)
print('The dimession of the test dataset is:', test.shape)


# In[3]:


train.info()


# In[4]:


train.head()


# In[5]:


train.describe()


# In[6]:


test.head()


# In[7]:


test.describe()


# > <div class="alert alert-block alert-info">
# <b>💡</b> There are no missing values neither in the train nor test datasets. </div>

# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Data Exploration</h1> 

# In[8]:


round(100*train['Class'].value_counts() / train.shape[0], 2) 


# In[9]:


sns.countplot(x = 'Class', hue = 'Class', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above chart, we see that the data is an extreme imbalanced (~99.8% are 0s and ~0.2% are 1s).</div>

# In[10]:


fig, axes = plt.subplots(2, 3, figsize = (20, 12))

sns.boxplot(ax = axes[0, 0], x = 'Class', y = 'Time', hue = 'Class', data = train)
sns.boxplot(ax = axes[0, 1], x = 'Class', y = 'V1', hue = 'Class', data = train)
sns.boxplot(ax = axes[0, 2], x = 'Class', y = 'V2', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 0], x = 'Class', y = 'V3', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 1], x = 'Class', y = 'V4', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 2], x = 'Class', y = 'V5', hue = 'Class', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above panels, most of features the have very similar distributions. However, when we compare the medians (of fraud and no-fraud observations) of V3 and V4, it seems that there is a small difference. 
# </div>

# In[11]:


fig, axes = plt.subplots(2, 3, figsize = (20, 12))

sns.boxplot(ax = axes[0, 0], x = 'Class', y = 'V6', hue = 'Class', data = train)
sns.boxplot(ax = axes[0, 1], x = 'Class', y = 'V7', hue = 'Class', data = train)
sns.boxplot(ax = axes[0, 2], x = 'Class', y = 'V8', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 0], x = 'Class', y = 'V9', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 1], x = 'Class', y = 'V10', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 2], x = 'Class', y = 'V11', hue = 'Class', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above panels, most of the features have very similar distributions. However, when we compare the medians (of fraud and no-fraud observations) of V9 and V11, it seems that there is a small difference. 
# </div>

# In[12]:


fig, axes = plt.subplots(2, 3, figsize = (20, 12))

sns.boxplot(ax = axes[0, 0], x = 'Class', y = 'V12', hue = 'Class', data = train)
sns.boxplot(ax = axes[0, 1], x = 'Class', y = 'V13', hue = 'Class', data = train)
sns.boxplot(ax = axes[0, 2], x = 'Class', y = 'V14', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 0], x = 'Class', y = 'V15', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 1], x = 'Class', y = 'V16', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 2], x = 'Class', y = 'V17', hue = 'Class', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above panels, most of the features have very similar distributions. However, when we compare the medians (of fraud and no-fraud observations) of V15, it seems that there is a small difference. 
# </div>

# In[13]:


fig, axes = plt.subplots(2, 3, figsize = (20, 12))

sns.boxplot(ax = axes[0, 0], x = 'Class', y = 'V18', hue = 'Class', data = train)
sns.boxplot(ax = axes[0, 1], x = 'Class', y = 'V19', hue = 'Class', data = train)
sns.boxplot(ax = axes[0, 2], x = 'Class', y = 'V20', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 0], x = 'Class', y = 'V21', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 1], x = 'Class', y = 'V22', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 2], x = 'Class', y = 'V23', hue = 'Class', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above panels, most of the features have very similar distributions. However, when we compare the medians (of fraud and no-fraud observations) of V18 and V19, it seems that there is a small difference. 
# </div>

# In[14]:


fig, axes = plt.subplots(2, 3, figsize = (20, 12))

sns.boxplot(ax = axes[0, 0], x = 'Class', y = 'V24', hue = 'Class', data = train)
sns.boxplot(ax = axes[0, 1], x = 'Class', y = 'V25', hue = 'Class', data = train)
sns.boxplot(ax = axes[0, 2], x = 'Class', y = 'V26', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 0], x = 'Class', y = 'V27', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 1], x = 'Class', y = 'V28', hue = 'Class', data = train)
sns.boxplot(ax = axes[1, 2], x = 'Class', y = 'Amount', hue = 'Class', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above panels, there is no much to learn because the distribution of the fraud and no-fraud groups are very similar. 
# </div>
# 
# We next proceed to explore potential correlation among the predictor features.

# In[15]:


## Explore the correlation between all numerical features
corr_mat = train.drop(columns = ['id', 'Class'], axis = 1).corr()

## Keep only correlation higher than a threshold
threshold = 0.3
corr_threshold = corr_mat[(corr_mat > threshold) | (corr_mat < -threshold)]

plt.figure(figsize = (12, 9))
sns.heatmap(corr_threshold, annot = True, cmap = 'seismic', fmt = ".2f",
            linewidths=.5, cbar_kws={'shrink': .5},annot_kws={'size': 8})
plt.title("Correlations Among Features")
plt.show();


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above heatmap, V2, V5 and V20 are the features the have the higher correlations with Amount. We also see that:
#     <ul>
#     <li> V1 and V3 have a correlation of 0.3. </li>
#     <li> V9 and V10 have a correlation of 0.31. </li>
#     <li> V9 and V10 have a correlation of 0.31. </li>
#     <li> V12 and V14 have a correlation of 0.31. </li>  
#     <li> V21 and V22 have a correlation of 0.30. </li>    
#     </ul>
# </div>
# 
# We next proceed to compare the distribution of the features in the train and test datasets.

# In[16]:


train_vis = train.drop(columns = 'Class', axis = 1).reset_index(drop = True).copy()
test_vis = test.copy()

train_vis['Dataset'] = 'Train'
test_vis['Dataset'] = 'Test'
data_tot = pd.concat([train_vis, test_vis], axis = 0).reset_index(drop = True)

fig, axes = plt.subplots(2, 3, figsize = (20, 12))

sns.kdeplot(ax = axes[0, 0], x = 'Time', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[0, 1], x = 'V1', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[0, 2], x = 'V2', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 0], x = 'V3', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 1], x = 'V4', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 2], x = 'V5', hue = 'Dataset', data = data_tot, fill = True);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above panels, we see that the distributions of Time are very different in the train and test datasets. Also the distribution of V3 is sligthly different in the train and test datasets.
# </div>

# In[17]:


fig, axes = plt.subplots(2, 3, figsize = (20, 12))

sns.kdeplot(ax = axes[0, 0], x = 'V6', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[0, 1], x = 'V7', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[0, 2], x = 'V8', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 0], x = 'V9', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 1], x = 'V10', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 2], x = 'V11', hue = 'Dataset', data = data_tot, fill = True);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above panels, we see that the distribution of the features and very similar in the train and test datasets.
# </div>

# In[18]:


fig, axes = plt.subplots(2, 3, figsize = (20, 12))

sns.kdeplot(ax = axes[0, 0], x = 'V12', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[0, 1], x = 'V13', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[0, 2], x = 'V14', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 0], x = 'V15', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 1], x = 'V16', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 2], x = 'V17', hue = 'Dataset', data = data_tot, fill = True);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above panels, we see that the distribution of the features and very similar in the train and test datasets.
# </div>

# In[19]:


fig, axes = plt.subplots(2, 3, figsize = (20, 12))

sns.kdeplot(ax = axes[0, 0], x = 'V18', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[0, 1], x = 'V19', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[0, 2], x = 'V20', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 0], x = 'V21', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 1], x = 'V22', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 2], x = 'V23', hue = 'Dataset', data = data_tot, fill = True);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above panels, we see that the distribution of the features and very similar in the train and test datasets.
# </div>

# In[20]:


fig, axes = plt.subplots(2, 3, figsize = (20, 12))

sns.kdeplot(ax = axes[0, 0], x = 'V24', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[0, 1], x = 'V25', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[0, 2], x = 'V26', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 0], x = 'V27', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 1], x = 'V28', hue = 'Dataset', data = data_tot, fill = True)
sns.kdeplot(ax = axes[1, 2], x = 'Amount', hue = 'Dataset', data = data_tot, fill = True);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above panels, we see that the distribution of the features and very similar in the train and test datasets.
# </div>

# <div class="alert alert-block alert-info">
# <b>💡 Insights from EDA:</b><br> 
# <ul>
#     <li> Extreme imbalanced dataset (~99.8% are 0s and ~0.2% are 1s). </li>
#     <li> In the train dataset, when we compare the medians (of fraud and no-fraud observations) of V3 and V4, it seems that there is a small difference.  </li>
#     <li> In the train dataset, when we compare the medians (of fraud and no-fraud observations) of V9 and V11, it seems that there is a small difference.  </li>
#     <li> In the train dataset, when we compare the medians (of fraud and no-fraud observations) of V15, it seems that there is a small difference.  </li>
#     <li> In the train dataset, when we compare the medians (of fraud and no-fraud observations) of V18 and V19, it seems that there is a small difference. </li>
#     <li> V2 and Amount have a correlation of 0.56. </li>
#     <li> V20 and Amount have a correlation of 0.53. </li>
#     <li> The distributions of Time are very different in the train and test datasets. </li>
#     <li> The distributions of V3 are slighlty different in the train and test datasets. </li>
# </ul>
# </div>

# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">XGBoost Modeling</h1> 

# In[21]:


X = train.drop(columns = ['id', 'Class'], axis = 1)
Y = train['Class']
test = test.drop(columns = 'id', axis = 1)

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
                               max_depth = 7, 
                               min_child_weight = 10, 
                               n_estimators = 1000, 
                               subsample = 0.7).fit(X_train, Y_train)
        XGB_imp.append(XGB_md.feature_importances_)
        
        ## Predicting on X_test and test
        XGB_pred_1 = XGB_md.predict_proba(X_test)[:, 1]
        XGB_pred_2 = XGB_md.predict_proba(test)[:, 1]
        
        ## Computing roc-auc score
        XGB_cv_scores.append(roc_auc_score(Y_test, XGB_pred_1))
        preds.append(XGB_pred_2)

XGB_cv_score = np.mean(XGB_cv_scores)    
print('The average roc-auc score over 5-folds (run 5 times) is:', XGB_cv_score)


# We next proceed to visualize the importance of the features under the XGBoost model. 

# In[22]:


plt.figure(figsize = (12, 14))
pd.DataFrame(XGB_imp, columns = X.columns).apply(np.mean, axis = 0).sort_values().plot(kind = 'barh');
plt.xlabel('XGBoost Score')
plt.ylabel('Feature')
plt.show();


# In[23]:


## Building model in the entire train dataset
XGB_md = XGBClassifier(tree_method = 'gpu_hist',
                       colsample_bytree = 0.7, 
                       gamma = 0.8, 
                       learning_rate = 0.01, 
                       max_depth = 7, 
                       min_child_weight = 10, 
                       n_estimators = 1000, 
                       subsample = 0.7).fit(X, Y)

train_preds = pd.DataFrame({'Class': Y, 'Class_pred': XGB_md.predict_proba(X)[:, 1]})
train_preds.head()


# In[24]:


RocCurveDisplay.from_predictions(train_preds['Class'], train_preds['Class_pred'])
plt.title('XGBoost ROC-AUC Curve on Train')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate');


# In[25]:


XGB_preds_train = train_preds['Class_pred']
XGB_preds_test = pd.DataFrame(preds).apply(np.mean, axis = 0)

data1 = pd.DataFrame({'Predicted Likelihood': XGB_preds_train})
data2 = pd.DataFrame({'Predicted Likelihood': XGB_preds_test})

fig, axes = plt.subplots(1, 2, figsize = (20, 8), dpi = 200)
sns.histplot(data = data1, kde = True, stat = 'density', ax = axes[0], log_scale = True).set(title = 'XGBoost Predicted Likelihood in Train (log-scale)')
sns.histplot(data = data2, kde = True, stat = 'density', ax = axes[1], log_scale = True).set(title = 'XGBoost Predicted Likelihood in Test (log-scale)')
plt.show();


# In[26]:


submission['Class'] = XGB_preds_test
submission.to_csv('XGB_submission.csv', index = False)


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">LightGBM Modeling</h1> 

# In[27]:


lgb_cv_scores, lgb_imp = list(), list()
preds = list()

## Running 5 times CV
for i in range(5):
    
    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
                
        ## Building RF model
        lgb_md = LGBMClassifier(n_estimators = 1000,
                                max_depth = 7,
                                learning_rate = 0.01,
                                num_leaves = 10,
                                lambda_l1 = 3,
                                lambda_l2 = 3,
                                bagging_fraction = 0.7,
                                feature_fraction = 0.7, 
                                device = 'gpu').fit(X_train, Y_train)
        lgb_imp.append(lgb_md.feature_importances_)
        
        ## Predicting on X_test and test
        lgb_pred_1 = lgb_md.predict_proba(X_test)[:, 1]
        lgb_pred_2 = lgb_md.predict_proba(test)[:, 1]
        
        ## Computing roc-auc score
        lgb_cv_scores.append(roc_auc_score(Y_test, lgb_pred_1))
        preds.append(lgb_pred_2)

lgb_cv_score = np.mean(lgb_cv_scores)    
print('The average oof roc-auc score over 5-folds (run 5 times) is:', lgb_cv_score)


# We next proceed to visualize the importance of the features under the LightGBM model. 

# In[28]:


plt.figure(figsize = (12, 14))
pd.DataFrame(lgb_imp, columns = X.columns).apply(np.mean, axis = 0).sort_values().plot(kind = 'barh');
plt.xlabel('LightGBM Score')
plt.ylabel('Feature')
plt.show();


# In[29]:


## Building model in the entire train dataset
lgb_md = LGBMClassifier(n_estimators = 1000,
                        max_depth = 7,
                        learning_rate = 0.01,
                        num_leaves = 10,
                        lambda_l1 = 3,
                        lambda_l2 = 3,
                        bagging_fraction = 0.7,
                        feature_fraction = 0.7, 
                        device = 'gpu').fit(X, Y)

train_preds = pd.DataFrame({'Attrition': Y, 'Attrition_pred': lgb_md.predict_proba(X)[:, 1]})
train_preds.head()


# In[30]:


RocCurveDisplay.from_predictions(train_preds['Attrition'], train_preds['Attrition_pred'])
plt.title('LightGBM ROC-AUC Curve on Train')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate');


# In[31]:


lgb_preds_train = train_preds['Attrition_pred']
lgb_preds_test = pd.DataFrame(preds).apply(np.mean, axis = 0)

data1 = pd.DataFrame({'Predicted Likelihood': lgb_preds_train})
data2 = pd.DataFrame({'Predicted Likelihood': lgb_preds_test})

fig, axes = plt.subplots(1, 2, figsize = (20, 8), dpi = 200)
sns.histplot(data = data1, kde = True, stat = 'density', ax = axes[0], log_scale = True).set(title = 'LightGBM Predicted Likelihood in Train (log-scale)')
sns.histplot(data = data2, kde = True, stat = 'density', ax = axes[1], log_scale = True).set(title = 'LightGBM Predicted Likelihood in Test (log-scale)')
plt.show();


# In[32]:


submission['Class'] = lgb_preds_test
submission.to_csv('LightGBM_submission.csv', index = False)


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Model Performance Comparisson</h1> 

# In[33]:


md_scores = pd.DataFrame({'XGB': XGB_cv_scores, 'LightGBM': lgb_cv_scores})
avg_scores = pd.DataFrame(md_scores.apply(np.mean, axis = 0))
# avg_scores['LB'] = [0.81891, 0.82954]
avg_scores.columns = ['Avg. OOF ROC-AUC Score (CV-Score)']
avg_scores


# In[34]:


plt.figure(figsize = (10, 8))

plt.boxplot(md_scores, labels = ['XGBoost', 'LightGBM'])
plt.xlabel('Model')
plt.ylabel('5-folds CV-Score');


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Feature Engineering</h1> 
# 
# First, we start by taking a close look at the `Time` feature. According to the data description, this feature contains the seconds elapsed between each transaction and the first transaction in the dataset. Let's explore the time distributions in hours.

# In[35]:


train['Time'] = train['Time'] / 3600
test['Time'] = test['Time'] / 3600

fig, axes = plt.subplots(1, 2, figsize = (15, 6))

sns.histplot(ax = axes[0], x = 'Time', bins = 10, fill = True, data = train).set_title('Time (in hours) after first transaction in Train')
sns.histplot(ax = axes[1], x = 'Time', bins = 10,  fill = True, data = test).set_title('Time (in hours) after first transaction in Test');


# In `Time` distribution in the train dataset, approx. 82\% of the transactions occurred between 7 hours and 25 hours after the first transaction. All these transactions are made by credit cards in September 2013 by European cardholders (according to the original dataset description). Assuming that most of the credit card transactions occur between 12 pm and midnight, we can map the transaction to time-buckets as follows:
# 
# | Time (in hours) | Time-Bucket |
# | --------------- | ----------- |
# | 0-4 | Early Morning |
# | 4-10 | Morning |
# | 10-16 | Afternoon |
# | 16-22 | Night |
# | 22-28 | Early Morning |
# | 28-34 | Morning |
# | 34-40 | Afternoon |
# | 40-46 | Night |
# | 47-53 | Early Morning | 
# 
# Using the above table, we can engineer a time feature. 

# In[36]:


train['Time_Label'] = np.where(((train['Time'] >= 0) & (train['Time'] < 4)), 'Early_Morning',
                               np.where(((train['Time'] >= 4) & (train['Time'] < 10)), 'Morning',
                                       np.where(((train['Time'] >= 10) & (train['Time'] < 16)), 'Afternoon',
                                                np.where(((train['Time'] >= 16) & (train['Time'] < 22)), 'Night',
                                                         np.where(((train['Time'] >= 22) & (train['Time'] < 28)), 'Early_Morning', 'Morning')))))

test['Time_Label'] = np.where(((test['Time'] >= 33) & (test['Time'] < 34)), 'Morning', 
                              np.where(((test['Time'] >= 34) & (test['Time'] < 40)), 'Afternoon', 
                                       np.where(((test['Time'] >= 40) & (test['Time'] < 46)), 'Night', 'Early_Morning')))

train_dummies = pd.get_dummies(train['Time_Label'])
test_dummies = pd.get_dummies(test['Time_Label'])

train = pd.concat([train, train_dummies], axis = 1)
train = train.drop(columns = ['Time_Label', 'Morning', 'Afternoon', 'Night'], axis = 1)

test = pd.concat([test, test_dummies], axis = 1)
test = test.drop(columns = ['Time_Label', 'Morning', 'Afternoon', 'Night'], axis = 1)


# In[37]:


X = train.drop(columns = ['id', 'Time', 'Class'], axis = 1)
Y = train['Class']

test = test.drop(columns = 'Time', axis = 1)

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
                               max_depth = 7, 
                               min_child_weight = 10, 
                               n_estimators = 1000, 
                               subsample = 0.7).fit(X_train, Y_train)
        XGB_imp.append(XGB_md.feature_importances_)
        
        ## Predicting on X_test and test
        XGB_pred_1 = XGB_md.predict_proba(X_test)[:, 1]
        XGB_pred_2 = XGB_md.predict_proba(test)[:, 1]
        
        ## Computing roc-auc score
        XGB_cv_scores.append(roc_auc_score(Y_test, XGB_pred_1))
        preds.append(XGB_pred_2)

XGB_cv_score = np.mean(XGB_cv_scores)    
print('The average roc-auc score over 5-folds (run 5 times) is:', XGB_cv_score)


# In[38]:


lgb_cv_scores, lgb_imp = list(), list()
preds = list()

## Running 5 times CV
for i in range(5):
    
    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
                
        ## Building RF model
        lgb_md = LGBMClassifier(n_estimators = 1000,
                                max_depth = 7,
                                learning_rate = 0.01,
                                num_leaves = 10,
                                lambda_l1 = 3,
                                lambda_l2 = 3,
                                bagging_fraction = 0.7,
                                feature_fraction = 0.7, 
                                device = 'gpu').fit(X_train, Y_train)
        lgb_imp.append(lgb_md.feature_importances_)
        
        ## Predicting on X_test and test
        lgb_pred_1 = lgb_md.predict_proba(X_test)[:, 1]
        lgb_pred_2 = lgb_md.predict_proba(test)[:, 1]
        
        ## Computing roc-auc score
        lgb_cv_scores.append(roc_auc_score(Y_test, lgb_pred_1))
        preds.append(lgb_pred_2)

lgb_cv_score = np.mean(lgb_cv_scores)    
print('The average oof roc-auc score over 5-folds (run 5 times) is:', lgb_cv_score)


# In[39]:


md_scores = pd.DataFrame({'XGB': XGB_cv_scores, 'LightGBM': lgb_cv_scores})
avg_scores = pd.DataFrame(md_scores.apply(np.mean, axis = 0))
avg_scores.columns = ['Avg. OOF ROC-AUC Score (CV-Score)']
avg_scores


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;"> Cross Validation (taking into account time)</h1> 
# 
# For more details see [here](https://www.kaggle.com/competitions/playground-series-s3e4/discussion/381415).

# In[40]:


train = train[(train['Time'] >= 10) & (train['Time'] <= 24)].reset_index(drop = True)

X = train.drop(columns = ['id', 'Time', 'Class'], axis = 1)
Y = train['Class']

# test = test.drop(columns = ['id', 'Time'], axis = 1)

XGB_cv_scores, XGB_imp = list(), list()
preds = list()

## Running 5 times CV
for i in range(5):
    
    skf = StratifiedKFold(n_splits = 5, shuffle = False)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
                
        ## Building RF model
        XGB_md = XGBClassifier(tree_method = 'gpu_hist',
                               colsample_bytree = 0.7, 
                               gamma = 0.72, 
                               learning_rate = 0.01, 
                               max_depth = 7, 
                               min_child_weight = 10, 
                               n_estimators = 1000, 
                               subsample = 0.7).fit(X_train, Y_train)
        XGB_imp.append(XGB_md.feature_importances_)
        
        ## Predicting on X_test and test
        XGB_pred_1 = XGB_md.predict_proba(X_test)[:, 1]
        XGB_pred_2 = XGB_md.predict_proba(test)[:, 1]
        
        ## Computing roc-auc score
        XGB_cv_scores.append(roc_auc_score(Y_test, XGB_pred_1))
        preds.append(XGB_pred_2)

XGB_cv_score = np.mean(XGB_cv_scores)    
print('The average roc-auc score over 5-folds (run 5 times) is:', XGB_cv_score)


# In[41]:


submission['Class'] = XGB_preds_test
submission.to_csv('XGB_time_submission.csv', index = False)

