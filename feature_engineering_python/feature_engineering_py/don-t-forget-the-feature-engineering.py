#!/usr/bin/env python
# coding: utf-8

# ##### Hey Dear Kagglers,
# ##### Wanted to put my 2 cents in;
# ##### While I see most of the public kernels on this competition focusing on parameter tunning/automl/cutting edge ML - I decided to show a slightly different approach -  and also a gentle reminder that the above is NOT a substitution for the good old Feature Engineering in any way.
# ##### So - what's happening here:
# * Some FE:
#      - One Hot Encoding + subsequent feature selection (see below)
#      - Target Encoding (generally very useful in case the test set is similar enough to your train (confirmed it is/adversarial validation))
# * SUPER SIMPLE Generalized Linear Model (Multinomial)  
# 
# ##### No fancy stuff here ... still performing better than most of the fancy stuff out there...
# ##### Stay tuned - more is to come (fancy stuff as well) ...
# ##### ... and Happy Kaggling :)

# #### Imports 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold
import gc


# #### Functions

# In[2]:


def target_encoder(class_='', smoothing=0.2, X_train=None, X_test=None):
    # Inspired by this great kernel - please upvote: https://www.kaggle.com/caesarlupum/2020-20-lines-target-encoding
    train = X_train.copy()
    train['target'] = np.where(tr['target']==class_, 1, 0)
    test = X_test.copy()
    train_y = train['target']
    train_id = train['id']
    test_id = test['id']
    train.drop(['target', 'id'], axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)
    
    cat_feat_to_encode = train.columns.tolist()
    
    oof = pd.DataFrame([])
    
    for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=42, shuffle=True).split(train, train_y):
        ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
        ce_target_encoder.fit(train.iloc[tr_idx, :], train_y.iloc[tr_idx])
        oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)
    
    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
    ce_target_encoder.fit(train, train_y)
    train = oof.reindex(train.index) #.sort_index()
    test = ce_target_encoder.transform(test)
    train.columns = [class_ + '_' + str(col) for col in train.columns]
    test.columns = [class_ + '_' + str(col) for col in test.columns]
    
    return train, test


# #### Read data

# In[3]:


tr = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/train.csv")
te = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/test.csv")
tr_te = pd.concat([tr, te], axis=0).reset_index(drop=True)


# #### One Hot Encoding

# In[4]:


enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
tr_te_ohe = pd.DataFrame(enc.fit_transform(tr_te.drop(['id','target'], axis=1).copy())).astype(int)
sel = VarianceThreshold(threshold=0.01) # remove sparse features
tr_te_ohe = pd.DataFrame(sel.fit_transform(tr_te_ohe))
tr_te_ohe.columns = ['ohe_' + str(col) for col in tr_te_ohe.columns]
tr_te_ohe = pd.concat([tr_te[['id','target']],tr_te_ohe], axis=1)


# #### Target Encoding

# In[5]:


tr_class_1, te_class_1 = target_encoder(class_='Class_1', X_train=tr, X_test=te)
tr_class_2, te_class_2 = target_encoder(class_='Class_2', X_train=tr, X_test=te)
tr_class_3, te_class_3 = target_encoder(class_='Class_3', X_train=tr, X_test=te)
tr_class_4, te_class_4 = target_encoder(class_='Class_4', X_train=tr, X_test=te)
tr_tgt = pd.concat([tr_class_1,tr_class_2,tr_class_3,tr_class_4], axis=1)
te_tgt = pd.concat([te_class_1,te_class_2,te_class_3,te_class_4], axis=1)
tr_te_tgt = pd.concat([tr_tgt, te_tgt], axis=0).reset_index(drop=True)


# #### Format out and cleanup

# In[6]:


tr_te_fin = pd.concat([tr_te_ohe, tr_te_tgt], axis=1)
tr_fin = tr_te_fin[tr_te_fin['target'].notnull()].drop(['id'], axis=1).copy()
te_fin = tr_te_fin[tr_te_fin['target'].isnull()].drop(['id','target'], axis=1).copy()


# In[7]:


del tr_class_1, te_class_1, tr_class_2, te_class_2, tr_class_3, te_class_3, tr_class_4, te_class_4, tr_te_ohe, tr_tgt, te_tgt, tr_te_tgt, tr_te_fin
gc.collect()


# #### Model

# In[8]:


# H2O ML MODEL ======================================================================================================================
# preproc ===========================
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# init ==============================
h2o.init(max_mem_size='8G')

# import data =======================
train = h2o.H2OFrame(tr_fin)
train["target"] = train["target"].asfactor()
test = h2o.H2OFrame(te_fin)

y = "target"
x = test.columns

# fit model =========================
glm_model = H2OGeneralizedLinearEstimator(
    family="multinomial", 
    solver='AUTO', 
    alpha=0.5,
    #lambda_=0.6,
    link='Family_Default',
    intercept=True,
    lambda_search=True, 
    nlambdas=100,
    max_iterations = 1000,
    #missing_values_handling='MeanImputation',
    standardize=True,
    nfolds = 5, 
    seed = 1333
)
glm_model.train(x=x, y=y, training_frame=train)


# In[9]:


# Eval mod ==========================
glm_model.model_performance(xval=True)


# In[10]:


# Model pred ========================
preds = glm_model.predict(test).as_data_frame()


# In[11]:


h2o.cluster().shutdown()


# #### Submission

# In[12]:


subm = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/sample_submission.csv")
subm = pd.concat([subm['id'], preds[['Class_1','Class_2','Class_3','Class_4']]],axis=1)
subm.to_csv("submission.csv", index=False)


# Hopefully you liked it (please upvote) - and saw something new today. Let me know if you have any questions in the comments :)
