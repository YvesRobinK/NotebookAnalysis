#!/usr/bin/env python
# coding: utf-8

# # ***[Disaster Tweets] Feature Engineering, EDA and Classification***

# <img src="https://miro.medium.com/max/1135/0*9GBBxsNvQhgGPyyW.jpg" width="500">

# # Import Libraries and Data

# In[1]:


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# In[2]:


train_df = pd.read_csv('../input/nlp-getting-started/train.csv')
train_df.head()


# In[3]:


test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
test_df.head()


# # EDA

# ### Common words used in Disaster Tweets

# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


from wordcloud import WordCloud

word_1 = '  '.join(list(train_df[train_df['target']==1]['text']))
word_1 = WordCloud(width=600, height=500).generate(word_1)
plt.figure(figsize=(13, 9))
plt.imshow(word_1)
plt.show()


# ### Common words used in Non-Disaster Tweets

# In[6]:


word_0 = '  '.join(list(train_df[train_df['target']==0]['text']))
word_0 = WordCloud(width=600, height=500).generate(word_0)
plt.figure(figsize=(13, 9))
plt.imshow(word_0)
plt.show()


# # Feature Engineering

# ### Drop unnessesary feature

# In[7]:


train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)


# ### Fill missing values

# In[8]:


miss_per = (train_df.isnull().sum()/len(train_df))*100
miss_per = miss_per.sort_values(ascending=False)

sns.barplot(x=miss_per.index, y=miss_per)
plt.xlabel('Features')
plt.ylabel('% of Missing Values')
plt.show()


# In[9]:


train_df['location'] = train_df['location'].fillna('None')
train_df['keyword'] = train_df['keyword'].fillna('None')
test_df['location'] = test_df['location'].fillna('None')
test_df['keyword'] = test_df['keyword'].fillna('None')


# ### Tokenization

# In[10]:


import nltk
from nltk import TweetTokenizer

tokenizer = TweetTokenizer()

train_df['tokens'] = [tokenizer.tokenize(item) for item in train_df.text]
test_df['tokens'] = [tokenizer.tokenize(item) for item in test_df.text]


# ### Lemmatization

# In[11]:


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize_item(item):
    new_item = []
    for x in item:
        x = lemmatizer.lemmatize(x)
        new_item.append(x)
    return " ".join(new_item)


# In[12]:


train_df['tokens'] = [lemmatize_item(item) for item in train_df.tokens]
test_df['tokens'] = [lemmatize_item(item) for item in test_df.tokens]


# ### Vectorization

# In[13]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
target = train_df['target']
train_df = train_df.drop('target', axis=1)
train_x_vec = vectorizer.fit_transform(train_df.tokens)
test_x_vec = vectorizer.transform(test_df.tokens)


# # Modeling

# In[14]:


X = train_x_vec
y = target


# ### Split train_df

# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)


# ### Define SVC model

# In[16]:


from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.svm import SVC

class_svc = SVC(probability=True, random_state=0)
class_svc.fit(X_train, y_train)
y_pred_svc = class_svc.predict(X_valid)

class_rep_svc = classification_report(y_valid, y_pred_svc)
print('\t\t\tClassification report:\n\n', class_rep_svc, '\n')

plot_confusion_matrix(class_svc, X_valid, y_valid)
plt.show()


# # Prediction

# In[17]:


class_svc = SVC(probability=True, random_state=0)
class_svc.fit(X, y)
pred = class_svc.predict(test_x_vec)


# In[18]:


submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
submission['target'] = pred
submission.to_csv('submission.csv', index=False)
submission


# # Reference Notebook
# [Tweets 🐦 : Disaster 💥 or Not ☔](https://www.kaggle.com/code/pralabhpoudel?kernelSessionId=90816622)
