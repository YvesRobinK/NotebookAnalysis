#!/usr/bin/env python
# coding: utf-8

# # Credit
# Fork from https://www.kaggle.com/code/x75a40890/ai-generated-text-detection-quick-baseline We wanted to check how RAPIDS models would fare.
# 
# Uses content from:
# https://www.kaggle.com/code/rsuhara/ai-generated-text-detection-quick-baseline
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
import umap


# # Importing files and Feature Engineering

# In[2]:


train = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")
extra_train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv")
extra_train


# In[3]:


extra_train = extra_train[extra_train.RDizzl3_seven]


# In[4]:


train.rename(columns={'generated':'label'}, inplace=True)

train = pd.concat([train, extra_train])
           
                          


# In[5]:


train


# In[6]:


test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')

df = pd.concat([train['text'], test['text']], axis=0)

vectorizer = TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True)
X = vectorizer.fit_transform(df)


# # Clustering

# In[7]:


embedding = umap.UMAP(random_state=2023, n_components=2).fit_transform(X)


# In[8]:


cl = KMeans(7)


# In[9]:


embeddings_human = embedding[:train.shape[0]]
embeddings_human = embeddings_human[train.label == 0]


# In[10]:


cl.fit(embeddings_human)
dist = cl.transform(embedding).min(1)
clusters = cl.predict(embedding)


# In[11]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


plt.scatter(embedding[:, 0], embedding[:, 1], marker='.', s=1, cmap='rainbow', c=clusters)


# In[13]:


plt.scatter(embedding[:train.shape[0]][train.label == 0 , 0], embedding[:train.shape[0]][train.label == 0, 1], marker='.', s=2)
plt.scatter(embedding[:train.shape[0]][train.label == 1 , 0], embedding[:train.shape[0]][train.label == 1, 1], marker='.', s=2)


# # Train Predictions

# In[14]:


preds_train = dist[:train.shape[0]]


# In[15]:


roc_auc_score(train.label, preds_train)


# In[16]:


bins = np.linspace(0, 10)
_ = plt.hist(preds_train[train.label == 0], bins=bins, alpha=0.5, log=True)
_ = plt.hist(preds_train[train.label == 1], bins=bins, alpha=0.5, log=True)


# # Test predictions

# In[17]:


preds_test = dist[train.shape[0]:]


# In[18]:


preds_test


# In[19]:


pd.DataFrame({'id':test["id"],'generated':preds_test}).to_csv('submission.csv', index=False)


# In[ ]:




