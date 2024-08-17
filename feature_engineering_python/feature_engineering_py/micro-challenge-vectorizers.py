#!/usr/bin/env python
# coding: utf-8

# # **Vectorizers** - #1 micro challenge
# # Rules
# I have an idea of an alternative challenge format for a while. I want to test it.
# In short, it's a short challenge with specific measurable goals to be achieved.
# 
# In this challenge, you are given a fixed pipeline and only can change the vectorization process. The vectorization method interface is fixed, the rest is up to you.
# 
# You need to **fork [original notebook](https://www.kaggle.com/dremovd/micro-challenge-vectorizers)**
# In order to compete, you also need to **make your Kaggle notebook public**.
# 
# # Challenge [data](https://www.kaggle.com/c/nlp-getting-started/data)
# Data is the same as for the official competition, you can read description here https://www.kaggle.com/c/nlp-getting-started/data
# 
# # Goals
# - 🥉 Bronze. F1-score >= **0.80** at **public** leaderboard 
# - 🥈 Silver. F1-score >= **0.81** at **public** leaderboard
# - 🥇 Gold. F1-score >= **0.81** at **public** leaderboard + runtime is below **1 minute**
# 
# # [Submit](https://forms.gle/H8MPo4xpu4NDVsX49)
# You can submit your **public** Kaggle notebook via this [link](https://forms.gle/H8MPo4xpu4NDVsX49) 
# # [Leaderboard](http://bit.ly/36pSp3S) 
# The final leaderboard is sorted by a medal type and then by submission time. The earlier you achieved the goal is better. You can see current leaderboard by this [link](http://bit.ly/36pSp3S)

# # Fixed pipeline
# In order to participate, the part below need to be unchanged

# In[1]:


import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
import scipy


def simple_pipeline():
    print("Load data")
    train, test = load_data()
    
    data = pd.concat([train, test], axis=0, ignore_index=True)
    print("Vectorization")
    X = vectorization(data.drop('target', axis=1))
    if type(X) == scipy.sparse.coo_matrix:
        X = X.tocsr()
        
    test_mask = data.is_test.values
    
    X_train = X[~test_mask]
    y_train = data['target'][~test_mask]
    
    X_test = X[test_mask]
    if scipy.sparse.issparse(X):
        X_train.sort_indices()
        X_test.sort_indices()

    model = build_model(X_train, y_train)
    
    print("Prediction with model")
    p = model.predict(X_test)
    
    print("Generate submission")
    make_submission(data[test_mask], p)


def load_data():
    train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
    train['is_test'] = False
    
    test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
    test['target'] = -1
    test['is_test'] = True
    
    return train, test


def calculate_validation_metric(model, X, y, metric):
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    score = cross_val_score(model, X, y, scoring=metric, cv=folds, n_jobs=4)
    
    return np.mean(score), model


def select_model(X, y):
    models = [
        LinearSVC(C=30),
        LinearSVC(C=10),
        LinearSVC(C=3),
        LinearSVC(C=1),
        LinearSVC(C=0.3),
        LinearSVC(C=0.1),
        LinearSVC(C=0.03),
        RidgeClassifier(alpha=30),
        RidgeClassifier(alpha=10),
        RidgeClassifier(alpha=3),
        RidgeClassifier(alpha=1),
        RidgeClassifier(alpha=0.3),
        RidgeClassifier(alpha=0.1),
        RidgeClassifier(alpha=0.03),
        LogisticRegression(C=30),
        LogisticRegression(C=10),
        LogisticRegression(C=3),
        LogisticRegression(C=1),
        LogisticRegression(C=0.3),
        LogisticRegression(C=0.1),
        LogisticRegression(C=0.03),
    ]
    
    results = [calculate_validation_metric(
        model, X, y, 'f1_macro',
    ) for model in models]

    best_result, best_model = max(results, key = lambda x: x[0]) 
    print("Best model validation result: {:.4f}".format(best_result))
    print("Best model: {}".format(best_model))
    
    return best_model


def build_model(X, y):
    print("Selecting best model")
    best_model = select_model(X, y)
    
    print("Refit model to full dataset")
    best_model.fit(X, y)
    
    return best_model

    
def make_submission(data, p):
    submission = data[['id']].copy()
    submission['target'] = p
    submission.to_csv('submission.csv', index=False)


# # Your part
# ## In *vectorization* method you can change everything and use any dependencies

# In[2]:


from sklearn.feature_extraction.text import CountVectorizer 

def vectorization(data):
    """
    data is concatenated train and test datasets with target excluded
    Result value "vectors" expected to have some number of rows as data
    """
    
    vectorizer = CountVectorizer()
    text = data['text'].fillna('')
    vectors = vectorizer.fit_transform(text)
    
    return vectors


# In[3]:


get_ipython().run_cell_magic('time', '', '\nsimple_pipeline()\n')

