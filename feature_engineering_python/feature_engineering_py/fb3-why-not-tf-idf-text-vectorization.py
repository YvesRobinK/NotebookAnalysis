#!/usr/bin/env python
# coding: utf-8

# # FB3 CTB with Term Frequency - Inverse Document Frequency (TF-IDF)
# 
# Bag of words (BoW) converts text into feature vector by # occurrences of words in an essay.
# 
# TF-IDF is based on BoW model, which contain insights about the importance of a word in the essay, and used to generate list of features for this notebook.
# 
# 
# ### **[WARN]** Why TF-IDF may not help in this competition?
# 
# **PROS:** Computationally cheap, understand the importance of words to a given document. Commonly used in building search engines, summarizing/classifying documents.
# 
# **CONS:** Cannot help carry semantic meaning of a word or a sentences and ignore word order, which is not recommended in this competition, as we are grading students' essay.
# 
# This is my first attempt on NLP and initially found that TF-IDF is great at text vectorization. After going through several rounds of code improvement and technical reading, I just discovered that TF-IDF may not be suitable for this task, as stated in CONS above.
# 
# I will keep this notebook as the baseline for TF-IDF learning and explain why it is not suitable for this task. I will continue to discover other method, such as BERT, DeBERTA.
# 
# Let me know in the comments, if you have any innovative ideas to improve based on TF-IDF approach.
# 
# ### List of References
# 
# CREDIT: Sheel Saket for Count Vectorizer vs TFIDF article
# 
# https://www.linkedin.com/pulse/count-vectorizers-vs-tfidf-natural-language-processing-sheel-saket/
# 
# CREDIT: Anirudha Simha for TF-IDF article
# 
# https://www.capitalone.com/tech/machine-learning/understanding-tf-idf/

# ## Import libraries

# In[267]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import string
import gc

#from wordcloud import WordCloud
#from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
#import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[268]:


train_df = pd.read_csv("/kaggle/input/feedback-prize-english-language-learning/train.csv")
test_df = pd.read_csv("/kaggle/input/feedback-prize-english-language-learning/test.csv")


# # Why TF-IDF may not help
# 
# I use only 5 words extracted from TfidfVectorizer to illustrate the problem. This may be resolved by obtaining full dictionary from train dataset, but if the test dataset does not contain the words in dictionary, then we will still get incorrect predictions.

# In[281]:


train_df['datatype'] = "train"
test_df['datatype'] = "test"
fulldata = pd.concat([train_df,test_df],axis=0)

def vectorize_text(data):
    vectorizer = TfidfVectorizer(ngram_range=(1,1),max_df=1,max_features=5)
    vec_text = vectorizer.fit_transform(data['full_text'])
    X = pd.DataFrame.sparse.from_spmatrix(vec_text)
    X.columns = vectorizer.get_feature_names_out().tolist()
    X = X.reset_index(drop=True)
    data = data.reset_index(drop=True)
    newdata = pd.concat([data, X],axis=1)
    del X
    gc.collect()
    return newdata

fulldata = vectorize_text(fulldata)


# In[299]:


test = fulldata[fulldata['datatype']=="test"]
train = fulldata[fulldata['datatype']=="train"]
TARGET = ["cohesion","syntax","vocabulary","phraseology","grammar","conventions"]
x = train.drop(columns=TARGET)
x = x.drop(columns=['text_id','full_text','datatype'])
testx = test.drop(columns=TARGET)
testx = testx.drop(columns=['text_id','full_text','datatype'])
y = train[TARGET]
print("Selected 5 features: ",x.columns.tolist())
print(testx)


# In[302]:


# Model training and prediction
regr = MultiOutputRegressor(CatBoostRegressor(verbose=False))
regr.fit(x,y)
pred = regr.predict(testx)
pred


# As test dataset does not contain words in the dictionary, their prediction score are same, which is why this approach may not help in this task.

# # Data preprocessing
# Basic feature list:
# 
# * number of characters
# * number of words
# * average length of words
# * number of stopwords
# * 1000 vectorized features
# 
# This is my first time in NLP, more feature to be added/optimized across time

# In[303]:


train_df = pd.read_csv("/kaggle/input/feedback-prize-english-language-learning/train.csv")
test_df = pd.read_csv("/kaggle/input/feedback-prize-english-language-learning/test.csv")


# In[334]:


def clean_text(text):
    text_nonum = re.sub(r'\d+','',text)
    text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation])
    text_no_doublespace = re.sub('\s+',' ',text_nopunct).strip()
    return text_no_doublespace

def count_stopwords(corpus):
    stop_words = set(stopwords.words('english'))
    count = 0 
    for word in corpus:
        if word in stop_words:
            count += 1
    return count

def vectorize_text(data):
    vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=10,max_features=100)
    vec_text = vectorizer.fit_transform(data['full_text'])
    X = pd.DataFrame.sparse.from_spmatrix(vec_text)
    X.columns = vectorizer.get_feature_names_out().tolist()
    X=X.reset_index(drop=True)
    data=data.reset_index(drop=True)
    newdata = pd.concat([data, X],axis=1)
    del X
    gc.collect()
    return newdata

def preprocessing(data):
    
    # Clean text
    data['full_text'] = data['full_text'].apply(lambda x: clean_text(x))
    
    # Feature engineering  
    data['num_char'] = data["full_text"].str.len()
    data['num_words'] = data["full_text"].apply(lambda x: len(x.split()))
    data['ave_word_length'] = data["full_text"].str.split().apply(lambda x: np.mean([len(i) for i in x]))
    data['corpus'] = data['full_text'].apply(lambda x: ''.join(x).split())
    data['num_stopwords'] = data['corpus'].apply(lambda x: count_stopwords(x))
    
    # Feature normalization
    data['num_char'] = data['num_char']/data["num_char"].max()
    data['num_words'] = data['num_words']/data["num_words"].max()
    data['ave_word_length'] = data['ave_word_length']/data["ave_word_length"].max()
    data['num_stopwords'] = data['num_stopwords']/data["num_stopwords"].max()
    
    # TfidfVectorizer
    data = vectorize_text(data)
    
    return data


# In[335]:


train_df['datatype'] = "train"
test_df['datatype'] = "test"
fulldata = pd.concat([train_df,test_df],axis=0)
fulldata = preprocessing(fulldata)
train = fulldata[fulldata['datatype']=="train"]
train = train.drop(columns=["full_text","corpus","text_id","datatype"])
test = fulldata[fulldata['datatype']=="test"]
test = test.drop(columns=["full_text","corpus","text_id","datatype"])


# In[336]:


TARGET = ["cohesion","syntax","vocabulary","phraseology","grammar","conventions"]
x = train.drop(columns=TARGET)
testx = test.drop(columns=TARGET)
y = train[TARGET]


# # CTB model training with CV
# * K-fold=5 cross validation for each target

# Thanks Ryan Li for MCRMSE function

# In[337]:


def mcrmse(targets, predictions):
    error = targets - predictions
    squared_error = np.square(error)
    colwise_mse = np.mean(squared_error, axis=0)
    root_colwise_mse = np.sqrt(colwise_mse)
    return np.mean(root_colwise_mse, axis=0)


# In[338]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)


# In[339]:


# GPU is recommended if feature size is large
regr = CatBoostRegressor(l2_leaf_reg=0.1,
                         early_stopping_rounds=10,
                         metric_period=500,
                         verbose=False)
predTrain = y_val[TARGET].copy()
predTest = test[TARGET].copy()
for feature in TARGET:
    cv = cross_validate(regr,X_train,y_train[feature], cv=2,return_train_score=True, return_estimator=True)
    pred = np.zeros(X_val.shape[0])
    predT = np.zeros(testx.shape[0])
    for estimator in cv['estimator']:
        pred += estimator.predict(X_val)
        predT += estimator.predict(testx)
    pred /= len(cv['estimator'])
    predT /= len(cv['estimator'])
    print("RMSE for feature "+ feature + " is " + str(mean_squared_error(y_val[feature],pred)))
    predTrain.loc[:,feature] = pred
    predTest.loc[:,feature] = predT
    
print("MCRMSE:",mcrmse(y_val,predTrain))


# In[340]:


testprediction = predTest.copy()
testprediction = testprediction.reset_index()


# # Final submission

# In[341]:


submitData = pd.read_csv('/kaggle/input/feedback-prize-english-language-learning/sample_submission.csv')
output = pd.DataFrame({'text_id': submitData.text_id, 
                       'cohesion': testprediction['cohesion'],
                       'syntax': testprediction['syntax'],
                       'vocabulary':testprediction['vocabulary'],
                       'phraseology':testprediction['phraseology'],
                       'grammar':testprediction['grammar'],
                       'conventions':testprediction['conventions']})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# In[342]:


output


# In[ ]:




