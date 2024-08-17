#!/usr/bin/env python
# coding: utf-8

# # Credit
# Fork from https://www.kaggle.com/code/xiaocao123/ai-generated-text-detection-add-new-data
# 
# Using new train dataset: https://www.kaggle.com/datasets/thedrcat/daigt-v3-train-dataset/

# # Importing library

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier


# # Importing files and Feature Engineering

# In[2]:


train = pd.read_csv("/kaggle/input/daigt-v3-train-dataset/train_v3_drcat_01.csv")
train = train.dropna(subset='text')
train = train[train.RDizzl3_seven == True].reset_index(drop=True)
test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
df = pd.concat([train['text'], test['text']], axis=0)
vectorizer = TfidfVectorizer(ngram_range=(1, 3),sublinear_tf=True)
X = vectorizer.fit_transform(df)


# In[3]:


train.value_counts("label")


# # Logistic Regression

# In[4]:


lr_model = LogisticRegression()


# # SGD

# In[5]:


sgd_model = SGDClassifier(max_iter=5000, tol=1e-3, loss="modified_huber")


# # Voting Classifier

# In[6]:


# Create the ensemble model
ensemble = VotingClassifier(estimators=[('lr', lr_model),('sgd', sgd_model)],weights=[0.1,0.9],voting='soft')
ensemble.fit(X[:train.shape[0]], train.label)


# In[7]:


preds_test = ensemble.predict_proba(X[train.shape[0]:])[:,1]


# In[8]:


pd.DataFrame({'id':test["id"],'generated':preds_test}).to_csv('submission.csv', index=False)

