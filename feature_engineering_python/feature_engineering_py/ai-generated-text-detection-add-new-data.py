#!/usr/bin/env python
# coding: utf-8

# 
# ## If this helps you, please give me a vote
# https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/456140
# # Credit
# Fork from
# 
# https://www.kaggle.com/code/rsuhara/ai-generated-text-detection-quick-baseline <br>
# https://www.kaggle.com/code/nahman/0-908-don-t-try-this-at-home
# 
# max_ngram 3 -> 5 :https://www.kaggle.com/code/chenbaoying/0-911-ai-generated-text-detection-test-feature<br>
# 
# Inspired by VLADIMIR DEMIDOV's work : <br>
# https://www.kaggle.com/code/yekenot/llm-detect-by-regression <br>
# https://www.kaggle.com/code/x75a40890/ai-generated-text-detection-quick-baseline <br>
# 
# Using new train dataset https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset/ <br>
# 

# # Notice
# 
# ## language_tool_python Has GPL-3.0 license, so not sure if it can be used.
# 
# ## I would use something else instead

# In[1]:


get_ipython().system('pip install -q language-tool-python --no-index --find-links ../input/daigt-misc/')
get_ipython().system('mkdir -p /root/.cache/language_tool_python/')
get_ipython().system('cp -r /kaggle/input/daigt-misc/lang57/LanguageTool-5.7 /root/.cache/language_tool_python/LanguageTool-5.7')


# # Importing library

# In[2]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import language_tool_python


# In[3]:


from concurrent.futures import ProcessPoolExecutor

tool = language_tool_python.LanguageTool('en-US')

def correct_sentence(sentence):
    return tool.correct(sentence)

# def how_many_typos(sentence):
#     return len(tool.check(sentence))

def correct_df(df):
    with ProcessPoolExecutor() as executor:
        df['text'] = list(executor.map(correct_sentence, df['text']))


# # Importing files and Feature Engineering

# In[4]:


# train = pd.read_csv("/kaggle/input/llm-7-prompt-training-dataset/train_essays_RDizzl3_seven_v2.csv")
train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv")
train1 = train[train.RDizzl3_seven == False].reset_index(drop=True)
train1=train[train["label"]==1].sample(8000)
train = train[train.RDizzl3_seven == True].reset_index(drop=True)
train=pd.concat([train,train1])
train['text'] = train['text'].str.replace('\n', '')

test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
test['text'] = test['text'].str.replace('\n', '')
correct_df(test)
df = pd.concat([train['text'], test['text']], axis=0)
vectorizer = TfidfVectorizer(ngram_range=(1, 5),sublinear_tf=True)
# X = vectorizer.fit_transform(df)
vectorizer = vectorizer.fit(test['text'])
X = vectorizer.transform(df)


# In[5]:


train.value_counts("label")


# # Logistic Regression

# In[6]:


lr_model = LogisticRegression()


# # SGD

# In[7]:


sgd_model = SGDClassifier(max_iter=5000, tol=1e-3, loss="modified_huber")


# # Voting Classifier

# In[8]:


# # Create the ensemble model
ensemble = VotingClassifier(estimators=[('lr', lr_model),('sgd', sgd_model)],weights=[0.01,0.99],voting='soft')
ensemble.fit(X[:train.shape[0]], train.label)


# In[9]:


preds_test = ensemble.predict_proba(X[train.shape[0]:])[:,1]


# In[10]:


pd.DataFrame({'id':test["id"],'generated':preds_test}).to_csv('submission.csv', index=False)


# In[ ]:




