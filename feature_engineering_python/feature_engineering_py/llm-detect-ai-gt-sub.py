#!/usr/bin/env python
# coding: utf-8

# * This is worth mentioning that the Public LB for this competition is highly overfitted since you can see the test set is so small. Even including a text preprocessing (which is a standard process) pipeline is decreasing the score on the LB. So I would like to advise all the people who are directly forking and making the submissions to not completely rely on these notebooks. Try making more robust models with text cleaning and preprocessing functions, better text encoders like Word2Vec, BERT and better models like LSTMs (Sequential) or GNNs (Graph-Based) so that you have a good score in the Private LB as well.
# 

# * The next few notebooks I'll publish will be having better models and text preprocessing pipelines. Just a heads up, I have made some submissions with a private notebook and the scores are not good, but we'll see that these notebooks will score higher on the Private LB.

# # Importing library

# In[1]:


get_ipython().system('pip install -q language-tool-python --no-index --find-links ../input/daigt-misc/')
get_ipython().system('mkdir -p /root/.cache/language_tool_python/')
get_ipython().system('cp -r /kaggle/input/daigt-misc/lang57/LanguageTool-5.7 /root/.cache/language_tool_python/LanguageTool-5.7')


# In[2]:


import numpy as np
import pandas as pd
import regex as re
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import make_scorer, accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import language_tool_python
from concurrent.futures import ProcessPoolExecutor
from sklearn.naive_bayes import MultinomialNB
seed = 202


# In[3]:


train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv")
external_train = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")
external_train.rename(columns={'generated': 'label'}, inplace=True)


# In[4]:


def seed_everything(seed=202):
    import random
    random.seed(seed)
    np.random.seed(seed)

seed_everything(seed)


# # Data Imports and Feature Engineering

# In[5]:


tool = language_tool_python.LanguageTool('en-US')
def correct_sentence(sentence):
    return tool.correct(sentence)
def correct_df(df):
    with ProcessPoolExecutor() as executor:
        df['text'] = list(executor.map(correct_sentence, df['text']))


# In[6]:


def how_many_typos(text):    
    return len(tool.check(text))


# In[7]:


not_persuade_df = train[train['source'] != 'persuade_corpus']
persuade_df = train[train['source'] == 'persuade_corpus']
sampled_persuade_df = persuade_df.sample(n=6000, random_state=42)

all_human = set(list(''.join(sampled_persuade_df.text.to_list())))
other = set(list(''.join(not_persuade_df.text.to_list())))
chars_to_remove = ''.join([x for x in other if x not in all_human])
print(chars_to_remove)

translation_table = str.maketrans('', '', chars_to_remove)
def remove_chars(s):
    return s.translate(translation_table)


# In[8]:


train=pd.concat([train,external_train])
train['text'] = train['text'].apply(remove_chars)
train['text'] = train['text'].str.replace('\n', '')

test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
test['text'] = test['text'].str.replace('\n', '')
test['text'] = test['text'].apply(remove_chars)
correct_df(test)
df = pd.concat([train['text'], test['text']], axis=0)


# In[9]:


vectorizer = TfidfVectorizer(ngram_range=(3, 5),tokenizer=lambda x: re.findall(r'[^\W]+', x),token_pattern=None,strip_accents='unicode',)
vectorizer = vectorizer.fit(test['text'])
X = vectorizer.transform(df)


# # Models

# In[10]:


lr=LogisticRegression()
clf = MultinomialNB(alpha=0.02)
sgd_model = SGDClassifier(max_iter=5000, tol=1e-3, loss="modified_huber")   
sgd_model2 = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber", class_weight="balanced") 
sgd_model3 = SGDClassifier(max_iter=10000, tol=5e-4, loss="modified_huber", early_stopping=True)


# # Voting Classifier

# In[11]:


ensemble = VotingClassifier(estimators=[('lr',lr),('mnb',clf),('sgd', sgd_model),('sgd2', sgd_model2),('sgd3', sgd_model3)],voting='soft')
ensemble.fit(X[:train.shape[0]], train.label)


# In[12]:


preds_test = ensemble.predict_proba(X[train.shape[0]:])[:,1]


# # Submission

# In[13]:


ntypos=test['text'].apply(lambda x: how_many_typos(x))
test['ntypos'] = -ntypos
test['generated'] = preds_test


# In[14]:


submission = pd.DataFrame({
    'id': test["id"],
    'generated': test['generated']
})
submission.to_csv('submission.csv', index=False)


# In[ ]:




