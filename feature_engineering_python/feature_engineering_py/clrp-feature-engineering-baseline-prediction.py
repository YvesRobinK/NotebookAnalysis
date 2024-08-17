#!/usr/bin/env python
# coding: utf-8

# ## Notebook Overview
# 
# The goal of the notebook is to clean CLRP Data, vectorize the excerpt data and add additional features to the data. The main packages used here are NLTK, regex, and pandas, and sklearn to achieve this.
# 
# Here are the various processes in this notebook:
# 
# 1. Reading Data
# 2. Clean the Data
# 3. Feature Engineering
# 4. Vectorize data
# 5. Build a model & create baseline predictions

# **Process 1: Reading the data**

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_error #Check the r2 error
from sklearn.metrics import r2_score #Check the r2 error
import numpy as np
from sklearn.preprocessing import MinMaxScaler #Perform data scaling
from sklearn.model_selection import cross_val_score, GridSearchCV #Cross valdiation scores

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Verify that the data import and explore the data by printing the top 3 rows in the dataframe.

# In[2]:


train_df=pd.read_csv(os.path.join(dirname, filenames[1]))
train_df.head(3)


# In[3]:


test_df=pd.read_csv(os.path.join(dirname, filenames[2]))
test_df.head(3)


# In[4]:


len(test_df)


# **Process 2: Clean the Data**
# 
# Convert the excerpt into lower case(so that an accurate count of the words can be obtained).
# Then remove the most frequently occurring words like - 'an', 'the', and 'on'. This list of frequently occurring can be obtained from the NLTK library.
# 
# **Process 3: Feature engineering**
# 
# Three new features were created:
# 
#     1.Average sentence length of the Excerpt 
#     2.Normalized word count
#     3.Normalized stopword frequency

# In[5]:


from nltk.corpus import stopwords
import spacy
import timeit
import re


nlp = spacy.load('en')
punct=";|!|:|;|,|-|'"
stop=set(stopwords.words('english'))

def preprocess_dataframe(df):
    #Set a unique Numbering for each exerpt
    df=df.reset_index()  
    #Average excerpt length
    train_df['excerpt_length']=train_df['excerpt'].str.len()
    avg_excerpt_len=train_df['excerpt_length'].mean().round(0) #Avg. excerpt length
    #Convert all text to lowecase
    df['excerpt_preprocess']=df['excerpt'].str.lower()         
    #FEATURE ENGINEERING: Get the legth of each excerpt
    df['excerpt_actual_length']=df['excerpt_preprocess'].str.len()
    #Remove common words from excerpt
    df['excerpt_preprocess']=df['excerpt_preprocess'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
    #FEATURE ENGINEERING: Get the legth of the preprocessed excerpt
    df['excerpt_preprocessed_length']=df['excerpt_preprocess'].str.len()
    #FEATURE ENGINEERING: Percent frequent words
    df['excerpt_stopword_freq']=(df['excerpt_actual_length']-df['excerpt_preprocessed_length'])/df['excerpt_actual_length']
    #FEATURE ENGINEERING: Get count of punctuations in the excerpt
    df['excerpt_punct_count']=df['excerpt'].apply(lambda x: len(re.findall(punct, x)))
    #Convert excerpt into setences
    df['excerpt_sentence'] = df['excerpt_preprocess'].apply(lambda x: list(nlp(x).sents))
    #Convert each setence of the exerpt into a pandas row
    df=df.explode('excerpt_sentence')
    #Convert spacy object to string object
    df['excerpt_sentence']=df['excerpt_sentence'].apply(lambda x: x.text)    
    ##FEATURE ENGINEERING: Get sentence length
    df['sentence_length']=df['excerpt_sentence'].str.len()
    ##FEATURE ENGINEERING: Get word count
    df['totalwords'] = df['excerpt_sentence'].str.split().map(len)
    ##FEATURE ENGINEERING: Get normalized word count
    df['normalized_word_count'] = round(df['sentence_length']/df['totalwords'],2)
    ##FEATURE ENGINEERING: Get normalized stopword frequency
    df['normalized_stopword_freq']=round(df['excerpt_stopword_freq']*avg_excerpt_len,1)
    ##FEATURE ENGINEERING: Get average senetence length
    df['avg sent length']=df[['sentence_length', 'index']].groupby(['index']).agg(['median'])
    ##FEATURE ENGINEERING: Get average senetence length
    df=df[['index','id','excerpt','excerpt_preprocess','avg sent length','normalized_word_count','normalized_stopword_freq']].drop_duplicates(subset ='index').set_index('index')
    return df


# Call the above function to - clean and generate features for the train data

# In[6]:


from datetime import datetime

now = datetime.now()
target=train_df['target']
train_df=preprocess_dataframe(train_df)
train_df['target']=target
later = datetime.now()
difference = int((later - now).total_seconds())

print("Execution Time: ",difference)
print("Dataframe length: ",len(train_df))


# In[7]:


train_df.head(3)


# In[8]:


train_df['target'].head()


# In[9]:


len(train_df)


# **Process 4: Vectorize the Data**
# 
# Convert the excerpt into a sparse matrix using TFIDF.

# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
x= v.fit_transform(train_df['excerpt'])
df1 = pd.DataFrame(x.toarray())
train_df_x=train_df[['avg sent length','normalized_word_count','normalized_stopword_freq']]
train_df_x = pd.concat([train_df_x, df1], axis = 1)


# **Normalize the data**

# In[11]:


scaler = MinMaxScaler()
train_df_x = scaler.fit_transform(train_df_x)


# In[12]:


train_df_x


# **Process 5: Model building & Prediction**
# 
# Define the baseline model for the prediction.

# In[13]:


from sklearn.ensemble import RandomForestRegressor

now = datetime.now()
regr = RandomForestRegressor(random_state=0)


# In[14]:


regr


# **Fit the model using the features and the target score**

# In[15]:


regr.fit(train_df_x, train_df['target'])


# In[16]:


#Get the predicted target scores
y_train_predict=regr.predict(train_df_x)


# **Get the rmse Score of the model.**

# In[17]:


#Cross validation r2 score
scores = cross_val_score(regr, y_train_predict.reshape(-1, 1), train_df['target'], cv=3, scoring='neg_root_mean_squared_error')
scores


# In[18]:


#r2_score(y_train_predict, train_df['target'])
round(np.sqrt(mean_squared_error(y_train_predict, train_df['target'])),3)


# In[19]:


r2_score(y_train_predict, train_df['target'])


# **Get the execution time of the prediction model**

# In[20]:


later = datetime.now()
difference = int((later - now).total_seconds())
print("Sklearn execution time: ",difference)


# **Preprocess the test data and get the test scores**

# In[21]:


now = datetime.now()
test_df=preprocess_dataframe(test_df)
later = datetime.now()
difference = int((later - now).total_seconds())

print("Execution Time: ",difference)
print("Dataframe length: ",len(test_df))


# In[22]:


x_test= v.transform(test_df['excerpt'])
df1 = pd.DataFrame(x_test.toarray())
test_df_x=test_df[['avg sent length','normalized_word_count','normalized_stopword_freq']]
test_df_x = pd.concat([test_df_x, df1], axis = 1)
test_df_x = scaler.transform(test_df_x)


# In[23]:


y_test_predict = regr.predict(test_df_x)
ids = test_df['id']

print(y_test_predict.shape)
print(type(y_test_predict))


# In[24]:


submission_df = pd.DataFrame({'id': ids, 'target': y_test_predict})
submission_df.to_csv('/kaggle/working/submission.csv', index=False)


# In[25]:


submission_df


# In[ ]:




