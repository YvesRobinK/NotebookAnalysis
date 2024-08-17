#!/usr/bin/env python
# coding: utf-8

# # RAPIDS cuML SVC Baseline
# 
# In this notebook we create a simple RAPIDS cuML SVC baseline. Using SVC works great for small datasets such as the train data in Kaggle's ICR - Identifying Age-Related Conditions competition because it resists overfitting. Also using a `for-loop`, we can engineer features (with cuDF) and evaluate them quickly because RAPIDS cuML SVC is so fast! After we find great features, we can then use them in heavier models such as LGBM DART or NN. Currently this notebook only has CV 0.44 but we can engineer features and boost its CV and LB!
# 
# UPDATE: In version 3 we found weights for the features but these weights overfit the train data and did poorly on LB. Now in version 4, we try something else. We will balance the classes during training. This improves the metric which is balanced log loss to **CV = 0.35**!
# 
# # Version Notes
# 
# * **Version 1** - has bug in CV computation code. The correct CV score is 0.44. Thank you @rohitsingh9990 @tuongkhang and @jhconnolly @vuxxxx @allmendinger for discovering this and posting comments. Now the CV score and LB score are more related.
# * **Version 2** - is baseline using correct CV computation code. CV = 0.44, LB = 0.35
# * **Version 3** - adds weights to the SVC features. We boost CV score to CV = 0.25. The LB score is 0.41. This didn't work.
# * **Version 4** - balances class weights during training with downsampling. We achieve CV score CV = 0.35. Let's see what LB is...

# # Load Libraries and Train Data
# Note that Kaggle's installation of RAPIDS cuML requires us to import cuDF first. Otherwise we get an error.

# In[1]:


import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import cudf
from cuml.svm import SVC
from sklearn.model_selection import KFold


# In[2]:


train = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
print('Train data shape', train.shape )
train.head()


# # Feature Engineer
# SVC does better if we standarize all the features and fillna. Note this preprocess procedure is also great for MLP. If we make an MLP we should predict both `Class` and all the targets in the `greeks.csv` file. Using auxiliary targets with multitask learning will help our model become smarter (on the task we care about). Also it is a good way to use data that is available for train but not test.

# In[3]:


train['EJ'] = train.EJ.map({'A':0,'B':1})

FEATURES = []
# SAVE MEANS, STDS, NANS FOR TEST INFERENCE LATER
means = {}; stds = {}; nans = {}

for c in train.columns[1:-1]:
    m = train[c].mean()
    means[c] = m
    s = train[c].std()
    stds[c] = s
    train[c] = (train[c]-m)/s
    n = train[c].min() - 0.5
    nans[c] = n
    train[c] = train[c].fillna(n)
    FEATURES.append(c)


# # Train RAPIDS SVC
# We will train RAPIDS SVC using simple KFold. I tried stratified KFold and the CV results are similar. We will use bagging and downsampling to balance the classes. By bagging we will help the model see all the data even though we are downsampling negatives. By balancing the classes we improve the competition metric which is `balanced log loss`.

# In[4]:


BAGS = 20
FOLDS = 11
oof = np.zeros(len(train))
models = {}

for bag in range(BAGS):
    print('#'*25)
    print('### Bag',bag+1)
    print('#'*25)
    models[bag] = []
    skf = KFold(n_splits=FOLDS, shuffle=True, random_state=bag)
    for fold,(train_idx, valid_idx) in enumerate(skf.split(X=train[FEATURES], y=train['Class'])):

        print('=> Fold',fold+1,', ',end='')

        # DOWNSAMPLE NEGATIVE CLASS TO BALANCE CLASSES
        y_train = train.loc[train_idx,'Class']
        RMV = y_train.loc[y_train==0].sample(
            frac=0.7, random_state=bag*BAGS+fold).index.values
        train_idx = np.setdiff1d(train_idx,RMV)

        # TRAIN DATA
        X_train = train.loc[train_idx,FEATURES]
        y_train = train.loc[train_idx,'Class']

        # VALID DATA
        X_valid = train.loc[valid_idx,FEATURES]
        y_valid = train.loc[valid_idx,'Class']

        # TRAIN MODEL
        clf = SVC(C=5, probability=True) 
        clf.fit(X_train, y_train)

        # RAPIDS SVC WILL RETURN PANDAS DATAFRAME
        oof[valid_idx] += clf.predict_proba(X_valid).iloc[:,1].values / BAGS
        models[bag].append(clf)
    print()


# # Compute CV Score
# Our baseline CV score is 0.44. We can improve it by engineering features and/or using feature selection.
#   
# **UPDATE:** Using balanced training, we achieve **CV = 0.35**, hooray!

# In[5]:


# https://www.kaggle.com/code/datafan07/icr-simple-eda-baseline
def balance_logloss(y_true, y_pred):
    
    y_pred = np.stack([1-y_pred,y_pred]).T
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
    y_pred / np.sum(y_pred, axis=1)[:, None]
    nc = np.bincount(y_true)
    
    logloss = (-1/nc[0]*(np.sum(np.where(y_true==0,1,0) * np.log(y_pred[:,0]))) - 1/nc[1]*(np.sum(np.where(y_true!=0,1,0) * np.log(y_pred[:,1])))) / 2
    
    return logloss

m = balance_logloss( train.Class.values, oof )
print('CV Score =',m)


# In[6]:


plt.hist(oof,bins=100)
plt.title('Histogram of OOF',size=20)
plt.show()


# # Infer Test
# The test data shape is 5 rows in this commit notebook, but when we submit to Kaggle then our code will load the hidden test data that has 400 rows and make predictions for both public and private test.

# In[7]:


test = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
print('Test data shape', test.shape )
test.head()


# # Feature Engineer
# We perform the exact same feature engineering that we did with train. Specifically we standardize using the same means, stds, and nans.

# In[8]:


test['EJ'] = test.EJ.map({'A':0,'B':1})
for c in test.columns[1:]:
    m = means[c]
    s = stds[c]
    test[c] = (test[c]-m)/s
    n = nans[c]
    test[c] = test[c].fillna(n)


# # SVC Inference

# In[9]:


preds = np.zeros(len(test))

for bag in range(BAGS):
    for clf in models[bag]:
            X_test = test[FEATURES]
            preds += clf.predict_proba(X_test).iloc[:,1].values / FOLDS / BAGS


# # Create Submission CSV

# In[10]:


sub = test[['Id']].copy()
sub['Class_0'] = 1-preds
sub['Class_1'] = preds
sub.to_csv('submission.csv',index=False)
sub.head()

