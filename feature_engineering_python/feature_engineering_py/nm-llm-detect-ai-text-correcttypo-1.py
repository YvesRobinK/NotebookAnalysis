#!/usr/bin/env python
# coding: utf-8

# # Credit
# 
# Dataset: Augmented dataset by @jdragonxherrera https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/456729
# 
# Fork from https://www.kaggle.com/code/xiaocao123/ai-generated-text-detection-add-new-data
# 
# https://www.kaggle.com/code/rsuhara/ai-generated-text-detection-quick-baseline
# 
# Inspired by VLADIMIR DEMIDOV's work : <br>
# https://www.kaggle.com/code/yekenot/llm-detect-by-regression <br>
# https://www.kaggle.com/code/x75a40890/ai-generated-text-detection-quick-baseline
# 
# Using new train dataset https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset/
# 
# @chenbaoying
# https://www.kaggle.com/code/chenbaoying/0-911-ai-generated-text-detection-test-feature#Voting-Classifier
# 
# And for all others who contributed through discussions/codes

# # Importing library

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import random,os,sys
from concurrent.futures import ProcessPoolExecutor
import re
import joblib


# In[2]:


random.seed(42)
np.random.seed(42)
debug=False
pseudo=False


# In[3]:


get_ipython().system('pip install -q language-tool-python --no-index --find-links ../input/daigt-misc/')
get_ipython().system('mkdir -p /root/.cache/language_tool_python/')
get_ipython().system('cp -r /kaggle/input/daigt-misc/lang57/LanguageTool-5.7 /root/.cache/language_tool_python/LanguageTool-5.7')
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')


# # Importing files and Feature Engineering

# In[4]:


def denoise_text(text):
    # Assuming 'tool' is defined elsewhere in your code
    corrected_text = tool.correct(text)
    return corrected_text

# Function to correct the 'text' column of a DataFrame or Series in parallel
def correct_df(input_data):
    if isinstance(input_data, pd.DataFrame):
        # If input is a DataFrame, correct the 'text' column
        with ProcessPoolExecutor() as executor:
            input_data['text'] = list(executor.map(denoise_text, input_data['text']))
    elif isinstance(input_data, pd.Series):
        # If input is a Series, correct the series
        with ProcessPoolExecutor() as executor:
            input_data = list(executor.map(denoise_text, input_data)) 
    return input_data


# In[5]:


# train1 = pd.read_csv("/kaggle/input/augmented-data-for-llm-detect-ai-generated-text/final_train.csv")
# train2 = pd.read_csv("/kaggle/input/augmented-data-for-llm-detect-ai-generated-text/final_test.csv")

# train=pd.concat([train1,train2])
# print(train.label.value_counts())


# In[6]:


# def get_traindata(correct=False):
#     train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv")
# #     train = train2.sample(100)

#     train1=train[train["label"]==1].sample(8050) 
#     train = train[train.RDizzl3_seven == True].reset_index(drop=True) 
#     train=pd.concat([train,train1])
#     train['text'] = train['text'].str.replace('\n', '') 
#     print(train.label.value_counts())
#     train1 = train.copy()     
    
#     if correct:
#         train = correct_df(train1) 
#     return train

# train = get_traindata() 


# In[7]:


train = pd.read_csv("/kaggle/input/ai-essay-detection-daigt-v2-dataset-with-typos/daigt-v2-train-dataset-with-typos.csv")
# train = pd.read_csv('/kaggle/input/ai-essay-detection-daigt-v2-dataset-with-typos/daigt-v2-train-dataset-without-typos.csv')
 
train['text'] = train['text'].str.replace('\n', '')
 
train.label.value_counts()


# In[8]:


## credit @siddhvr
not_persuade_df = train[train['source'] != 'persuade_corpus']
persuade_df = train[train['source'] == 'persuade_corpus']
sampled_persuade_df = persuade_df.sample(n=10000, random_state=42)
# Testing idea from discussion with @nbroad about limited characters in human essays
all_human = set(list(''.join(sampled_persuade_df.text.to_list())))
other = set(list(''.join(not_persuade_df.text.to_list())))
chars_to_remove = ''.join([x for x in other if x not in all_human])
print(chars_to_remove)

translation_table = str.maketrans('', '', chars_to_remove)
def remove_chars(s):
    return s.translate(translation_table) 
train['text'] = train['text'].apply(remove_chars)


# In[9]:


get_ipython().run_cell_magic('time', '', '\nif debug:\n    test = pd.read_csv(\'/kaggle/input/augmented-data-for-llm-detect-ai-generated-text/final_test.csv\')\n    test = test.sample(1000)\n    test_labels = test.label\n    test.text[0]="appile is good for healt"\n    print("validation df",\'\\n\',test.label.value_counts())\nelse:\n    test = pd.read_csv(\'/kaggle/input/llm-detect-ai-generated-text/test_essays.csv\')\n\ntest[\'text\'] = test[\'text\'].str.replace(\'\\n\', \'\') \ntest[\'text\'] = test[\'text\'].apply(remove_chars) \n# correct_df(test) \n')


# ### Functions to vectorize and fit models

# In[10]:


get_ipython().run_cell_magic('time', '', '\ndef vectorize(train,test):\n    df = pd.concat([train[\'text\'], test[\'text\']], axis=0)     \n     \n    vectorizer1 = TfidfVectorizer(sublinear_tf=True,\n                             ngram_range=(3, 4),\n#                              tokenizer=lambda x: re.findall(r\'[^\\W]+\', x), \n                             token_pattern = r\'(?u)\\b\\w+\\b|\\b\\w\\b\', \n                             strip_accents=\'unicode\')                             \n    \n    vectorizer1.fit(test.text)    \n    X1 = vectorizer1.transform(df)\n\n    vectorizer2 = TfidfVectorizer(sublinear_tf=True,\n                             ngram_range=(3, 6),\n                             tokenizer=lambda x: re.findall(r\'[^\\W]+\', x),\n                             token_pattern=None,\n                             strip_accents=\'unicode\')\n    vectorizer2.fit(test.text)\n    X2 = vectorizer2.transform(df)\n    \n    vectorizer3 = TfidfVectorizer(sublinear_tf=True,\n                             ngram_range=(2,5),\n                             tokenizer=lambda x: re.findall(r\'[^\\W]+\', x),\n                             token_pattern=None,\n                             strip_accents=\'unicode\')\n    vectorizer3.fit(test.text)\n    X3 = vectorizer3.transform(df)\n    \n    return X1,X2,X3,vectorizer1,vectorizer2,vectorizer3\n    \n\n\ndef fit_predict(X1,X2,X3,train,final=False):\n    \n    sgd_model1 = SGDClassifier(max_iter=8000, tol=1e-3, loss="modified_huber") \n    sgd_model1.fit(X1[:train.shape[0]], train.label)\n    preds_test1 = sgd_model1.predict_proba(X1[train.shape[0]:])[:,1]\n    print("SGD Model 1 predictions completed")\n#     print("sgd model1 predictions \\n",preds_test1,\'\\n\')\n    \n    \n    mnb = MultinomialNB(alpha=0.02)\n    mnb.fit(X2[:train.shape[0]], train.label)\n    preds_test2 = mnb.predict_proba(X2[train.shape[0]:])[:,1]\n    print("MNB Model predictions completed")\n#     print("mnb predictions \\n",preds_test2,\'\\n\')\n\n#     sgd_model2 = SGDClassifier(max_iter=10000, tol=1e-3, loss="modified_huber",class_weight="balanced") \n#     sgd_model2.fit(X2[:train.shape[0]], train.label)\n#     preds_test2 = sgd_model2.predict_proba(X2[train.shape[0]:])[:,1]\n\n    sgd_model3 = SGDClassifier(max_iter=15000, tol=5e-4, loss="modified_huber",early_stopping=True) \n    sgd_model3.fit(X3[:train.shape[0]], train.label)\n    preds_test3 = sgd_model3.predict_proba(X3[train.shape[0]:])[:,1]\n    print("SGD Model 3 predictions completed")\n#     print("sgd model3 predictions \\n",preds_test3,\'\\n\')\n\n    preds_test = np.average([preds_test1,preds_test2,preds_test3],axis=0,weights=[0.6,0.2,0.2])\n#     print("ensemble predictions \\n",preds_test)\n    \n    if debug==True:\n        auc = roc_auc_score(test_labels,preds_test)\n        print("AUC Score for ensemble",auc)\n    \n    if final:\n        joblib.dump(sgd_model1, \'sgd_model1.joblib\')\n        joblib.dump(mnb, \'mnb.joblib\')\n        joblib.dump(sgd_model3, \'sgd_model3.joblib\')    \n    \n    return preds_test,sgd_model1,mnb,sgd_model3, \n\n')


# ### Vectorize

# In[11]:


X1,X2,X3,vectorizer1,vectorizer2,vectorizer3 = vectorize(train,test)


# ### Predict

# In[12]:


preds_test,sgd_model1,mnb,sgd_model3=fit_predict(X1,X2,X3,train)
# AUC Score for ensemble 0.9910720486111111


# In[13]:


if debug==True:
    auc = roc_auc_score(test_labels,preds_test)
    print("AUC Score for ensemble",auc)


# ## Pseudo-labelling

# In[14]:


if pseudo:    
    test['label']=preds_test
    test['text'] = test.apply(lambda row: correct_df(row['text']) if row['label'] >=0.5 else row['text'], axis=1)
    train['text'] = train.apply(lambda row: correct_df(row['text']) if row['label'] >=0.5 else row['text'], axis=1)
    test.drop(['label'],inplace=True,axis=1)
    X1,X2,X3,vectorizer1,vectorizer2,vectorizer3 = vectorize(train,test) 
    preds_test,sgd_model1,mnb,sgd_model3=fit_predict(X1,X2,X3,train,final=True)
    if debug==True:
        auc = roc_auc_score(test_labels,preds_test)
        print("AUC after pseudo-label based corrections \n",auc)


# ## Submit

# In[15]:


if debug==False:
    pd.DataFrame({'id':test["id"],'generated':preds_test}).to_csv('submission.csv', index=False)
    pd.read_csv('/kaggle/working/submission.csv')
    sub=pd.read_csv('/kaggle/working/submission.csv')
    print(sub)


# # External df test

# In[16]:


# # valid = pd.read_csv('/kaggle/input/ai-essay-detection-daigt-v2-dataset-with-typos/daigt-v2-test-dataset-with-typos-introduced.csv')
# # valid = pd.read_csv('/kaggle/input/ai-essay-detection-daigt-v2-dataset-with-typos/daigt-v2-train-dataset-without-typos.csv')
# valid = pd.read_csv('/kaggle/input/ai-essay-detection-daigt-v2-dataset-with-typos/daigt-v2-train-dataset-with-typos.csv')
# print(valid.label.value_counts())


# In[17]:


# df = valid.sample(1000)
# df['text'] = df['text'].str.replace('\n', '') 
# df['text'] = df['text'].apply(remove_chars)
# df1 = vectorizer1.transform(df.text)
# preds_valid =sgd_model1.predict_proba(df1)[:,1] 
# auc = roc_auc_score(df.label,preds_valid)
# print(preds_valid)
# print(auc)


# In[18]:


# def predict_validdf(df,model1,model2,model3,v1,v2,v3):
    
#     df1 = v1.transform(df.text)
#     df2 = v2.transform(df.text)
#     df3 = v3.transform(df.text)
    
#     preds_test1 = model1.predict_proba(df1)[:,1] 
#     preds_test2 = model2.predict_proba(df2)[:,1] 
#     preds_test3 = model3.predict_proba(df3)[:,1]   

#     preds_test = np.average([preds_test1,preds_test2,preds_test3],axis=0,weights=[0.6,0.2,0.2])
#     auc = roc_auc_score(df.label,preds_test)
#     print(auc)
#     return preds_test,auc

# preds_test_validdf,auc = predict_validdf(valid,sgd_model1,mnb,sgd_model3,vectorizer1,vectorizer2,vectorizer3)


# In[19]:


# from sklearn.model_selection import cross_validate
# cv_results = cross_validate(ensemble, X_train,y_train, cv=5)
# sorted(cv_results.keys())
# print(cv_results['test_score'])
# cv = cv_results['test_score']
# print(sum(cv)/len(cv))

