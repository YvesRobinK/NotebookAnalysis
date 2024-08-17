#!/usr/bin/env python
# coding: utf-8

# # Credit
# Fork from https://www.kaggle.com/code/rsuhara/ai-generated-text-detection-quick-baseline
# 
# Inspired by VLADIMIR DEMIDOV's work : <br>
# https://www.kaggle.com/code/yekenot/llm-detect-by-regression
# 
# For the training data we shall use the "RDizzl3 seven" dataset (v1) which can be found in the "LLM: 7 prompt training dataset" https://www.kaggle.com/datasets/carlmcbrideellis/llm-7-prompt-training-dataset
# 
# 

# # Importing library

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier


# # Importing files and Feature Engineering

# In[2]:


external_df = pd.read_csv("/kaggle/input/daigt-external-dataset/daigt_external_dataset.csv", sep=',')
print(external_df.shape)
external_df = external_df.rename(columns={'generated': 'label'})
external_df = external_df[["source_text"]]
external_df.columns = ["text"]
external_df['text'] = external_df['text'].str.replace('\n', '')
external_df["label"] = 1


# In[3]:


train = pd.read_csv("/kaggle/input/llm-7-prompt-training-dataset/train_essays_RDizzl3_seven_v1.csv")
train=pd.concat([train,external_df])
test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')

df = pd.concat([train['text'], test['text']], axis=0)

vectorizer = TfidfVectorizer(ngram_range=(1, 3),sublinear_tf=True)
X = vectorizer.fit_transform(df)


# # Logistic Regression

# In[4]:


lr_model = LogisticRegression(solver="liblinear")


# # SGD

# In[5]:


sgd_model = SGDClassifier(max_iter=1000, tol=1e-3, loss="modified_huber")


# # Voting Classifier

# In[6]:


# Create the ensemble model
ensemble = VotingClassifier(estimators=[('lr', lr_model),('sgd', sgd_model)], voting='soft')
ensemble.fit(X[:train.shape[0]], train.label)


# In[7]:


preds_test = ensemble.predict_proba(X[train.shape[0]:])[:,1]


# In[8]:


pd.DataFrame({'id':test["id"],'generated':preds_test}).to_csv('submission.csv', index=False)

