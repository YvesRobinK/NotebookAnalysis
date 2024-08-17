#!/usr/bin/env python
# coding: utf-8

# <center><h2 style='color:red'>NLP: The Simplest Way<br><span>By Kassem@elcaiseri</span></h2></center>
# <h3>NLP with Disaster Tweets (NLTK + Sklearn)</h3>
# * **1. Introduction**
# * **2. Data Preparation**
# * **3. Text Processing**
# * **4. Machine learning**
# * **5. Evaluate the model**
# * **5. Prediction and Submition**
# * **6. References**
# <hr>
# 
# * Update: using **word_tokenize()** rather than **text.split()**

# # 1. Introduction
# Twitter has become an important communication channel in times of emergency.
# The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).<br>
# **Goal** .. Predict which Tweets are about real disasters and which ones are not

# # 2. Data preparation

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


BASE = "/kaggle/input/nlp-getting-started/"
train = pd.read_csv(BASE + "train.csv")
test = pd.read_csv(BASE + "test.csv")
sub = pd.read_csv(BASE + "sample_submission.csv")


# In[4]:


tweets = train[['text', 'target']]
tweets.head()


# In[5]:


tweets.target.value_counts()


# In[6]:


tweets.shape


# # 3. Text Processing
# 

# In[7]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ## Remove Punctuation

# In[8]:


def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)


# In[9]:


tweets['text'] = tweets['text'].apply(remove_punctuation)
tweets.head(10)


# ## Remove Stopwords

# In[10]:


# extracting the stopwords from nltk library
sw = stopwords.words('english')
# displaying the stopwords
np.array(sw);


# In[11]:


def stopwords(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in word_tokenize(text) if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)


# In[12]:


tweets['text'] = tweets['text'].apply(stopwords)
tweets.head(10)


# ## Stemming operations
# Stemming operation bundles together words of same root. E.g. stem operation bundles "response" and "respond" into a common "respon

# In[13]:


# create an object of stemming function
stemmer = PorterStemmer()

def stemming(text):    
    '''a function which stems each word in the given text'''
    text = [stemmer.stem(word) for word in word_tokenize(text)]
    return " ".join(text) 


# In[14]:


tweets['text'] = tweets['text'].apply(stemming)
tweets.head(10)


# In[15]:


vectorizer = CountVectorizer(analyzer='word', binary=True, stop_words='english')
vectorizer.fit(tweets['text'])


# In[16]:


X = vectorizer.transform(tweets['text']).todense()
y = tweets['target'].values
X.shape, y.shape


# # 4. Machine learning

# In[17]:


from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)


# In[19]:


model = LogisticRegression(C=1.0, random_state=111)
model.fit(X_train, y_train)


# # 4. Evaluate the model

# In[20]:


y_pred = model.predict(X_test)

f1score = f1_score(y_test, y_pred)
print(f"Model Score: {f1score * 100:.2f} %")


# As you can see score imporved from 75% in last version to 77% with simple preprocessing change.

# # 5. Prediction and Submition

# In[21]:


tweets_test = test['text']
test_X = vectorizer.transform(tweets_test).todense()
test_X.shape


# In[22]:


lr_pred = model.predict(test_X)


# In[23]:


sub['target'] = lr_pred
sub.to_csv("submission.csv", index=False)
sub.head()


# # 6. References
# * https://www.kaggle.com/elcaiseri/toxicity-bias-logistic-regression-tfidfvectorizer
# * http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# * https://www.kaggle.com/itratrahman/nlp-tutorial-using-python/notebook

# <h3>Thanks For Being Here. <span style='color:red'>UPVOTE</span> If Interested .. Feel Free In Comments</h3>

# In[ ]:




