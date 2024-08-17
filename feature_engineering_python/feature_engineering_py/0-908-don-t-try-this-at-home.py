#!/usr/bin/env python
# coding: utf-8

# ## Improvement 
# 
# Fit the tf-idf only on test data
# 
# For more info, see this Discussion:  https://www.kaggle.com/code/nahman/0-908-don-t-try-this-at-home

# # Credit
# Fork from https://www.kaggle.com/code/rsuhara/ai-generated-text-detection-quick-baseline
# 
# Inspired by VLADIMIR DEMIDOV's work : <br>
# https://www.kaggle.com/code/yekenot/llm-detect-by-regression <br>
# https://www.kaggle.com/code/x75a40890/ai-generated-text-detection-quick-baseline
# 
# Using new train dataset https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset/

# # Importing library

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier


# # Importing files and Feature Engineering

# In[ ]:


# train = pd.read_csv("/kaggle/input/llm-7-prompt-training-dataset/train_essays_RDizzl3_seven_v2.csv")
train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv")
train1 = train[train.RDizzl3_seven == False].reset_index(drop=True)
train1=train[train["label"]==1].sample(8000)
train = train[train.RDizzl3_seven == True].reset_index(drop=True)
train=pd.concat([train,train1])
train['text'] = train['text'].str.replace('\n', '')

test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
df = pd.concat([train['text'], test['text']], axis=0)

vectorizer = TfidfVectorizer(ngram_range=(1, 3),sublinear_tf=True)
vectorizer = vectorizer.fit(test['text'])
X = vectorizer.transform(df)


# In[ ]:


train.value_counts("label")


# # Logistic Regression

# In[ ]:


lr_model = LogisticRegression()


# # SGD

# In[ ]:


sgd_model = SGDClassifier(max_iter=5000, tol=1e-3, loss="modified_huber")


# # Voting Classifier

# In[ ]:


# Create the ensemble model
ensemble = VotingClassifier(estimators=[('lr', lr_model),('sgd', sgd_model)],weights=[0.01,0.99],voting='soft')
ensemble.fit(X[:train.shape[0]], train.label)


# In[ ]:


preds_test = ensemble.predict_proba(X[train.shape[0]:])[:,1]


# In[ ]:


pd.DataFrame({'id':test["id"],'generated':preds_test}).to_csv('submission.csv', index=False)


# In[ ]:




