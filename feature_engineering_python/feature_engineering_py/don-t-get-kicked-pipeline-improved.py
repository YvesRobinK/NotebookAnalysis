#!/usr/bin/env python
# coding: utf-8

# Ref Articles 
# - https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
# - https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
# - https://discuss.analyticsvidhya.com/t/what-is-the-difference-between-predict-and-predict-proba/67376
# - https://github.com/AnilBetta/AV-Janata-Hack-healh-Care-2/blob/master/av-jh-hca2-cat.ipynb
# - https://github.com/gcspkmdr/HA-Hackathon

# ## Import Libraries

# In[60]:


import pandas as pd
import numpy as np
#from catboost import CatBoostClassifier
#from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
#from sklearn.metrics import accuracy_score

#Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

#For Missing Value and Feature Engineering
from sklearn.feature_selection import SelectKBest, chi2, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import time


# ## Import Data

# In[61]:


train = pd.read_csv("../input/DontGetKicked/training.csv")
test = pd.read_csv("../input/DontGetKicked/test.csv")


# In[62]:


train.head()


# ## SMOTE

# In[63]:


#insert code


# ## Feat Engineering

# In[64]:


# Date
#PurchDate


# In[65]:


train['mean_MMRAcquisitionAuctionAveragePrice_Make']=train.groupby(['Make'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
train['mean_MMRAcquisitionAuctionAveragePrice_Model']=train.groupby(['Model'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
train['mean_MMRAcquisitionAuctionAveragePrice_Trim']=train.groupby(['Trim'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
train['mean_MMRAcquisitionAuctionAveragePrice_SubModel']=train.groupby(['SubModel'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
train['mean_MMRAcquisitionAuctionAveragePrice_Color']=train.groupby(['Color'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
train['mean_MMRAcquisitionAuctionAveragePrice_Transmission']=train.groupby(['Transmission'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')


# In[66]:


test['mean_MMRAcquisitionAuctionAveragePrice_Make']=test.groupby(['Make'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
test['mean_MMRAcquisitionAuctionAveragePrice_Model']=test.groupby(['Model'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
test['mean_MMRAcquisitionAuctionAveragePrice_Trim']=test.groupby(['Trim'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
test['mean_MMRAcquisitionAuctionAveragePrice_SubModel']=test.groupby(['SubModel'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
test['mean_MMRAcquisitionAuctionAveragePrice_Color']=test.groupby(['Color'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
test['mean_MMRAcquisitionAuctionAveragePrice_Transmission']=test.groupby(['Transmission'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')


# ## Divide Dataset into X and Y

# In[67]:


#create X and y datasets for splitting 
X = train.drop(['RefId', 'IsBadBuy'], axis=1)
y = train['IsBadBuy']


# In[68]:


all_features = X.columns


# In[69]:


all_features = all_features.tolist()


# In[70]:


numerical_features = [c for c, dtype in zip(X.columns, X.dtypes)
                     if dtype.kind in ['i','f'] and c !='PassengerId']
categorical_features = [c for c, dtype in zip(X.columns, X.dtypes)
                     if dtype.kind not in ['i','f']]


# In[71]:


numerical_features


# In[72]:


categorical_features


# In[73]:


#import train_test_split library
from sklearn.model_selection import train_test_split

# create train test split
X_train, X_test, y_train, y_test = train_test_split( X,  y, test_size=0.3, random_state=0)  


# ## Setup Pipeline 

# In[95]:


preprocessor = make_column_transformer(
    
    (make_pipeline(
    #SimpleImputer(strategy = 'median'),
    KNNImputer(n_neighbors=2, weights="uniform"),
    MinMaxScaler()), numerical_features),
    
    (make_pipeline(
    SimpleImputer(strategy = 'constant', fill_value = 'missing'),
    OneHotEncoder(categories = 'auto', handle_unknown = 'ignore')), categorical_features),
    
)


# In[102]:


preprocessor_best = make_pipeline(preprocessor, 
                                  VarianceThreshold(), 
                                  SelectKBest(f_classif, k = 50)
                                 )


# In[103]:


RF_Model = make_pipeline(preprocessor_best, RandomForestClassifier(n_estimators = 100))


# ## Grid Search

# In[105]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 50)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
#Maximum number of levels in tree
max_depth = [2,4,6,8]
# Minimum number of samples required to split a node
#min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
#bootstrap = [True, False]


# In[106]:


# Create the param grid
param_grid = {'randomforestclassifier__n_estimators': n_estimators,
               'randomforestclassifier__max_features': max_features,
               'randomforestclassifier__max_depth': max_depth
               #'randomforestclassifier__min_samples_split': min_samples_split,
               #'randomforestclassifier__min_samples_leaf': min_samples_leaf,
               #'randomforestclassifier__bootstrap': bootstrap
             }
print(param_grid)


# In[118]:


from sklearn.model_selection import RandomizedSearchCV
rf_RandomGrid = RandomizedSearchCV(estimator = RF_Model, param_distributions = param_grid, cv = 3, verbose=1, n_jobs = -1, n_iter = 5, scoring = 'f1')


# In[119]:


rf_RandomGrid.fit(X_train, y_train)


# In[120]:


rf_RandomGrid.best_estimator_


# ## Accuracy

# In[121]:


print(f'Train : {rf_RandomGrid.score(X_train, y_train):.3f}')
print(f'Test : {rf_RandomGrid.score(X_test, y_test):.3f}')


# ## Gini Index

# In[122]:


def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


# In[123]:


actual_train = y_train
pred_train = rf_RandomGrid.predict(X_train)
actual_test = y_test
pred_test = rf_RandomGrid.predict(X_test)


# In[124]:


print(f'Gini Train : {gini(actual_train,pred_train):.3f}')
print(f'Gini Test : {gini(actual_test,pred_test):.3f}')


# ## Submission 

# In[125]:


test_pred = rf_RandomGrid.predict_proba(test[X.columns])[:,1]


# In[126]:


AllSub = pd.DataFrame({ 'RefId': test['RefId'],
                       'IsBadBuy' : test_pred
    
})


# In[127]:


AllSub['IsBadBuy'] = AllSub['IsBadBuy'].apply(lambda x: 1 if x > 0.09 else 0)


# In[128]:


AllSub.to_csv('DGK_RF_Pipe_BetterPipe1.csv', index = False)


# In[ ]:




