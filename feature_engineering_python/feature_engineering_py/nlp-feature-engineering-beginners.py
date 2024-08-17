#!/usr/bin/env python
# coding: utf-8

# **Hello Friends,
#  In this kernel my main aim is to make you guys familar with basic nlp techniques and feature engineering with codes and theory.**

# # Problem Statement

# Twitter has become an important communication channel in times of emergency.
# The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).
# 
# But, it’s not always clear whether a person’s words are actually announcing a disaster
# 
# > In this competition, you’re challenged to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t
# 
# Read more about this here --> https://www.kaggle.com/c/nlp-getting-started

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
print("Shape of train dataset is", train.shape)
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
print("Shape of test dataset is", test.shape)


# In[3]:


# with below method we can display maximum number of rows and columns we want to display.

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
train.head(10)


# # Check for target distribution

# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

def bar_plot(feature):
    sns.set(style="darkgrid")
    ax = sns.countplot(x=feature , data=train)
    
print("Total number of different target categories is", train.target.value_counts().count())
count_0 = train.target.value_counts()[0]
count_1 = train.target.value_counts()[1]
print("target with count 1 is {}".format(count_1))
print("target with count 0 is {}".format(count_0))
bar_plot("target")


# In[5]:


print("Total different categories in keyword is :", train.keyword.value_counts().count())
print("Total different categories in location is :", train.location.value_counts().count())


# # Checking for null values 

# > it always recommended to check total number of null values
# 
# > and then we have to decide where we should delete all the null rows or repalce by mean, median, mode or total number of counts

# In[6]:


train.isna().sum()


# > From the above we can see that  location contains 2533 null values followed by keywords with 61 null values

# # Before diving deep into text data lets explore categorical data

# > Checking not null locations, below I have limited it to 40, you can try will all

# In[7]:


train[~train["location"].isna()]["location"].tolist()[0:40]


# # Extract country from location

# > here i am making an extra column named country using geopy,
# 
# >  you can play with this geopy library to get latitude and longitude also..
# 
# > Comment down how you want to use geopy for this competion ?

# **let's extract the country name from given location**

# In[8]:


import geopy
import numpy as np
import pycountry

from geopy.geocoders import Nominatim
geolocator = Nominatim("navneet")
def get_location(region=None):
 
    if region:
        try:    
            return geolocator.geocode(region)[0].split(",")[-1] 
        except:
            return region
    return None

train["country"] = train["location"].apply(get_location)


# In[9]:


train[~train["country"].isna()]["country"].tolist()[30:50]


# In[10]:


train[~train["country"].isna()]["country"].nunique()


# > there are 86 unique country in dataframe.

# In[11]:


train[~train["country"].isna()]["country"].head()


# # Let's play with keyword

# In[12]:


set(train[~train["keyword"].isna()]["keyword"].tolist())


# > In the keywords we can see that few of the words are concatenated with "%20". Let's seperate these words

# In[13]:


def split_keywords(keyword):
    try:
        return keyword.split("%20")
    except:
        return [keyword]
    

train["keyword"] = train["keyword"].apply(split_keywords)


# In[14]:


train[~train["keyword"].isna()]["keyword"].tolist()[100:110]


# # Function to check if keywords exist in text or not

# In[15]:


def count_keywords_in_text(keywords, text):
    if not keywords[0]:
        return 0
    count = 0
    for keyword in keywords:
        each_keyword_count = text.count(str(keyword))
        count = count + each_keyword_count
    return count

train["keyword_count_in_text"] = train.apply(lambda row: count_keywords_in_text(row["keyword"] , row['text']), axis=1)


# In[16]:


train.tail()


# **future pending work to be done below**

# # Let's start doing analysis on text data

# > Analysing first 100 rows

# In[17]:


train["text"].tolist()[0:100]


# > form above we can see that we have #, ==>, ... and a lot of unnecessary words like to, is, are [stopwords], links that needs to be removed

# > In the below codes we are removing all the website links starting with http: or https:

#  # Count number of #(hash) in a text

# In[18]:


def get_count_of_hash(text):
    if not text:
        return -1
    return text.count("#")

train["count_#"] = train["text"].apply(get_count_of_hash)


#  # Count number of @(at the rate) in a text

# In[19]:


def get_count_of_at_rate(text):
    if not text:
        return -1
    return text.count("@")

train["count_@"] = train["text"].apply(get_count_of_at_rate)


# In[20]:


train["count_@"].to_list()[100:110]


# In[21]:


train.head()


# > Since this is twitter text, so counting number of hashes becomes more important

# **Remove website links**

# In[22]:


import re

print("Before---------")
print(train["text"].tolist()[31])

train['text'] = train['text'].str.replace('http:\S+', '', case=False)
train['text'] = train['text'].str.replace('https:\S+', '', case=False)
print("After----------")
print(train["text"].tolist()[31])


# > this way we can remove all website links

# **Removing all punctuations except hash**

# > punctuations should be removed because it doesnot add much value 

# In[23]:


import string
exclude = set(string.punctuation)
exclude_hash = {"#"}
exclude = exclude - exclude_hash
print("Length of punctuations to be excluded :",len(exclude))

print("Before---------")
print(train["text"].tolist()[0])

for punctuation in exclude:
  train['text'] = train['text'].str.replace(punctuation, '', regex=True)

print("After----------")
print(train["text"].tolist()[0])


# # Removing all the stop words

# > here i am adding stop words from two different package.
# 
# > you can check all the stop words by running below code.

# In[24]:


import nltk
nltk.download('stopwords')
from stop_words import get_stop_words
from nltk.corpus import stopwords

stop_words = list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english')) #About 179 stopwords
stop_words = sorted(set(stop_words).union(set(nltk_words)) - exclude_hash)  # removing hash from stop words

print("total stop words to be removed :", len(stop_words))


# In[25]:


print("Before--------")
print(train["text"].tolist()[0])
preprocessed_text = []
# tqdm is for printing the status bar
for sentance in train['text'].values:
    sent = ' '.join(e for e in sentance.split() if e not in stop_words)
    preprocessed_text.append(sent.lower().strip())

train["text"] = preprocessed_text
print("After----------")
print(train["text"].tolist()[0])


# > from above we can see that all the stop words like [are, of, this] has been removed.

# # Since dataset is very less so lets create out own w2v embedding

# **Lemmatise the words with spacy**

# **Why lemmatisation ?**
# 
# > lemmatisation is done on text data to get the lemma of that word.. Ex : stops -- > stop

# In[26]:


import spacy

# Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
nlp = spacy.load('en', disable=['parser', 'ner'])

print("Before--------")
print(train["text"].tolist()[2])

lemet_text = []
# tqdm is for printing the status bar
for sentance in train['text'].values:
    sent = " ".join([token.lemma_ for token in nlp(sentance)])
    lemet_text.append(sent.lower().strip())

train["text"] = lemet_text

train["text"] = lemet_text
print("After----------")
print(train["text"].tolist()[2])


# **Toekenizing the data**

# > tokenization is needed for making w2v models.
# 
# > "my name is navneet" --> after tokenization --> ["my", "name", "is", "navneet"]

# In[27]:


nltk.download('punkt')

train['text'] = train.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
train["text"].tolist()[2]


# **converting the data into vector forms using Word2Vec with vector size of 300**

# In[28]:


from gensim.models import Word2Vec
# train model
model = Word2Vec(train.text.values, min_count=1, size = 300)

# summarize vocabulary
words = list(model.wv.vocab)
#print(words)

# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)


# In[29]:


print(model.most_similar('disaster', topn = 20))


# # Work in progress, please upvote this kernel if you like my work and comment if i made any mistake.

# **Future work**
# 
# > different types of text embedding like countvectorizer, tfidf etc.
# 
# > some more feature engineering and cleaning
# 
# > differnt types of models like naive bayes, logistic, lightgbm
