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
# [**0.898**] Copy the weights=[0.21052631578947367, 0.7894736842105263] from https://www.kaggle.com/code/yongsukprasertsuk/ai-generated-text-mod-weight-add-more-data/notebook  
# 
# [**0.899**] Copy the weights=[0.15, 0.85] from https://www.kaggle.com/code/pamin2222/votingclf-alldata-modweight-0-899
# 
# [**0.901**] Remove LR: https://www.kaggle.com/code/rayljr/ai-generated-text-mod-weight-add-more-data-0-901
# 
# [**0.903**] Replace data and using 3 SGD, reference: https://www.kaggle.com/code/yongsukprasertsuk/ai-generated-text-mod-weight-add-more-data-0-902
# 
# [**0.911**] Only using test set to extract feature from TfidfVectorizer, Inspired by the discussion in: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/455701, My discussion: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/455997
# 
# [**0.917**] Remove typos in test.csv, reference: https://www.kaggle.com/code/xiaocao123/ai-generated-text-detection-add-new-data and https://www.kaggle.com/code/murugesann/nm-llm-detect-ai-text-typo-correct-with-testdata, Not reaching 0.918 may be the reason why I fix the random seed or use a different training set.
# 
# [**0.928**] ngram=(3,7) and Using new train set: https://www.kaggle.com/code/yongsukprasertsuk/ai-generated-text-mod-weight-add-more-data-0-928
# 
# [**0.930**] ngram=(3,5), reference:https://www.kaggle.com/code/hubert101/0-930-phrases-are-keys
# 
# [**0.931**] add new data: https://www.kaggle.com/code/jdragonxherrera/using-the-augmented-data **or** n_gram=(3,4) https://www.kaggle.com/code/yekenot/llm-detect-by-regression
# 
# [**0.932**] add new data: https://www.kaggle.com/code/jdragonxherrera/using-the-augmented-data **and** n_gram=(3,4) https://www.kaggle.com/code/yekenot/llm-detect-by-regression, the same hyperparameters setting I got **0.933** in my private notebook, maybe because of the random seed.
# 
# [**0.936**] new parameters for Classifiers（add MultinomialNB）: reference: https://www.kaggle.com/code/siddhvr/llm-detect-ai-gt-sub
# 
# [**0.938**] 3SGD+3MNB, Inspired by https://www.kaggle.com/code/siddhvr/llm-detect-ai-gt-sub
# 
# | Score |Dataset for TfidfVectorizer |ngram| Correct Typos | New Dataset | Classifiers |
# | --- | --- | --- | --- | --- | --- |
# |  0.903 | train + test | (1,5) | False |False | 3SGD |
# |  0.908 | train([original](/kaggle/input/llm-detect-ai-generated-text/train_essays.csv)) + test | (1,5) | False | False | 3SGD |
# |  0.871 | train | (1,5) | False | False | 3SGD |
# |  0.911 | test | (1,5) | False | False | 3SGD |
# |  0.519 | train[label=1] | (1,5) | False | False | 3SGD |
# |  0.917 | test | (1,5) | True | False | 3SGD |
# |  0.928 | test | (3,7) | True | True | 3SGD |
# |  0.930 | test | (3,5) | True | False | 3SGD |
# |  0.931 | test | (3,4) | True | False | 3SGD |
# |  0.932 | test | (3,4) | True | True  | 3SGD |
# |  0.935 | test | (3,4) | True | False  | 3SGD + MNB |
# |  0.927 | test | (3,4) | True | False  | 3MNB |
# |  0.938 | test | (3,4) | True | False  | 3SGD + 3MNB |
# 
# 
# ### try but not work:
# - ngram=(1,8)
# - correct typos of train set
# - five fold 

# # Importing library

# In[1]:


get_ipython().system('pip install -q language-tool-python --no-index --find-links ../input/daigt-misc/')
get_ipython().system('mkdir -p /root/.cache/language_tool_python/')
get_ipython().system('cp -r /kaggle/input/daigt-misc/lang57/LanguageTool-5.7 /root/.cache/language_tool_python/LanguageTool-5.7')


# In[2]:


import numpy as np
import pandas as pd
import time 
import os
import regex as re
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from scipy.sparse import csr_matrix, vstack, hstack
import language_tool_python
from concurrent.futures import ProcessPoolExecutor


# Setting hyperparameters
seed = 202 # set the seed for reproducibility  2023->202
isFixTestLeakage = False # TODO!!! reference: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/455701
isCorrectSentence = True


# # Seed Everything & Correct Sentence

# In[3]:


def seed_everything(seed=2023):
    import random
    random.seed(seed)
    np.random.seed(seed)

seed_everything(seed)

tool = language_tool_python.LanguageTool('en-US')
def correct_sentence(sentence):
    return tool.correct(sentence)

def how_many_typos(sentence):
    return len(tool.check(sentence))

def correct_df(df, max_workers=None):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        df['text'] = list(executor.map(correct_sentence, df['text']))


# # Importing files and Feature Engineering

# In[4]:


# # Using New data: https://www.kaggle.com/code/yongsukprasertsuk/ai-generated-text-mod-weight-add-more-data-0-928/notebook
# train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv")
# # train_data = train[train.RDizzl3_seven == False].reset_index(drop=True)
# train_data = train[train["label"]==1].sample(8000)
# train = train[train.RDizzl3_seven == True].reset_index(drop=True)
# train = pd.concat([train,train_data])
# train['text'] = train['text'].str.replace('\n', '')
# print(train.value_counts("label"))

# new data, reference:https://www.kaggle.com/code/jdragonxherrera/using-the-augmented-data
train = pd.read_csv("/kaggle/input/augmented-data-for-llm-detect-ai-generated-text/final_train.csv")
train = pd.concat((pd.read_csv("/kaggle/input/augmented-data-for-llm-detect-ai-generated-text/final_test.csv"), train))
print(f"{train.value_counts('label')}\n{train.head()}")

test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
test['text'] = test['text'].str.replace('\n', '')
# Correct typos of sentence
if isCorrectSentence:
    print("before:\n", test.head())
    correct_df(test)  # denoise
    print("After:\n", test.head())


# # Extract Feature

# In[5]:


# Extract Feature
min_ngram = 3  # 1 -> 3 -> 5
max_ngram = 4  # 3 -> 5 -> 7 -> 5 -> 4
vectorizer = TfidfVectorizer(ngram_range=(min_ngram, max_ngram),sublinear_tf=True)
if not isFixTestLeakage:
    df = pd.concat([train['text'], test['text']], axis=0)
    # X = vectorizer.fit_transform(df)
    vectorizer.fit_transform(test['text'])
    X = vectorizer.transform(df)
    print(X.shape)
else:
    # fix the test data leak, refenrece: https://www.kaggle.com/code/pamilovedl/ai-generated-text-detection-quick-baseline-0
    X = vectorizer.fit_transform(train['text'])  # only using train set
    X_test = vectorizer.transform(test['text'])
    print(X.shape, X_test.shape)


# # Creating Models

# In[6]:


def create_models(random_state=None):
    # Logistic Regression
    # lr_model = LogisticRegression(solver="liblinear")
    # SGD
    sgd_model = SGDClassifier(max_iter=5000, tol=1e-3, loss="modified_huber", 
                              random_state=random_state)  
    sgd_model2 = SGDClassifier(max_iter=5000, tol=1e-3, loss="modified_huber", 
                               random_state=(random_state + 1000) if random_state is not None else None, 
                               class_weight="balanced") 
    sgd_model3 = SGDClassifier(max_iter=10000, tol=5e-4, loss="modified_huber", 
                               random_state=(random_state + 2000) if random_state is not None else None, 
                               early_stopping=True)  
    # SVC
    # svc_model = SVC(probability=True)
    # MNB 
    mnb_model = MultinomialNB(alpha=0.02)
    mnb_model2 = MultinomialNB(alpha=0.1)
    mnb_model3 = MultinomialNB(alpha=0.2)
    # Voting Classifier
    estimators=[
        ('sgd1', sgd_model), 
        ('sgd2', sgd_model2),
        ('sgd3', sgd_model3),
        ('mnb1', mnb_model),
        ('mnb2', mnb_model2),
        ('mnb3', mnb_model3),
    ]
    # Create the ensemble model
    ensemble = VotingClassifier(
        estimators=estimators,
    #     weights=weights,
        voting='soft',
        verbose=0,
    )
    
    return ensemble


# # Training & Inference

# In[7]:


ensemble = create_models()
if not isFixTestLeakage:
    ensemble.fit(X[:train.shape[0]], train.label)
    preds_test = ensemble.predict_proba(X[train.shape[0]:])[:,1]
else:
    ensemble.fit(X, train.label)
    preds_test = ensemble.predict_proba(X_test)[:,1]


# # Submission

# In[8]:


sub = pd.DataFrame({'id':test["id"],'generated':preds_test})
sub.to_csv('submission.csv', index=False)
sub.head()

