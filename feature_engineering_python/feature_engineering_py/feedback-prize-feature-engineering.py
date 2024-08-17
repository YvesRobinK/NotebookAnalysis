#!/usr/bin/env python
# coding: utf-8

# <div style="color:#f56342;margin:0;font-size:40px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;overflow:hidden;font-weight:600;">Feedback Prize: <br>Feature Engineering & Data Construction</div>
# 
# <dim align='center'>
# <figure>
#   <img src="https://www.blumeglobal.com/wp-content/uploads/2018/11/NLP-image-scaled.jpg" alt="Trulli" style="width=70%">
# </figure>
# </dim>
# 

# <h5 style="text-align: center; font-family: Verdana; font-size: 12px; font-style: normal; font-weight: bold; text-decoration: None; text-transform: none; letter-spacing: 1px; color: #f56342; background-color: #fffff;">CREATED BY: Jonathan Bown</h5>

# <span style="font-size:16px; font-family:Verdana;">The dataset used to build models for my submissions for the most recent feedback prize is constructed with the code below. Various libraries are used to build the features and they incorporate varying degrees of time/space complexity.   </span>

# In[1]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
from textblob import TextBlob
import spacy
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from string import punctuation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
trans = preprocessing.MinMaxScaler()
from collections import Counter
nlp = spacy.load("en_core_web_lg")


# In[2]:


train=pd.read_csv('../input/feedback-prize-english-language-learning/train.csv')
test=pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')
display(train.head(2))
display(test.head(2))


# In[3]:


train.shape


# In[4]:


test.shape


# # Essay Examination

# In[5]:


train['full_text'].loc[4]


# In[6]:


train['full_text'].loc[1]


# # Text Feature Engineering

# ## Capital Letters

# In[7]:


def count_capital_words(text):
    return sum(map(str.isupper,text.split()))


# In[8]:


get_ipython().run_cell_magic('time', '', "var = 'n_capital'\ntrain[var]=train['full_text'].apply(count_capital_words)\ntest[var]=test['full_text'].apply(count_capital_words)\n")


# ## Punctuation

# In[9]:


def count_punctuations(text):
    punctuations="'!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'"
    count=0
    for i in punctuations:
        count+=text.count(i)
    return count


# In[10]:


get_ipython().run_cell_magic('time', '', "var = 'n_punct'\ntrain[var]=train['full_text'].apply(count_punctuations)\ntest[var]=test['full_text'].apply(count_punctuations)\n")


# In[11]:


#BASIC TEXT CLEANING
def text_cleaner(text):
    text = text.strip()
    text = re.sub(r'\n', ' ', text)
    text = text.lower()
    return text


# In[12]:


train['full_text']=train['full_text'].apply(text_cleaner)
test['full_text']=test['full_text'].apply(text_cleaner)


# ## Number of Unique Words

# In[13]:


def n_unique_words(text):
    text = text.translate(str.maketrans("", "", punctuation))
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    unique_words = np.unique(words)
    return len(unique_words)


# In[14]:


get_ipython().run_cell_magic('time', '', "var = 'n_unique'\ntrain[var]=train['full_text'].apply(n_unique_words)\ntest[var]=test['full_text'].apply(n_unique_words)\n")


# In[15]:


plt.figure(figsize=(5,20))
sns.displot(data=train,x=var,kde="True")
plt.title('Unique Word Count Distribution')
plt.show()


# ## Number of Unique Words (exclude stop words)

# In[16]:


def n_unique_words_no_stop(text):
    text = text.translate(str.maketrans("", "", punctuation))
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))
    #remove stop words
    words = [word for word in words if word not in stop_words]
    unique_words = np.unique(words)
    return len(unique_words)


# In[17]:


get_ipython().run_cell_magic('time', '', "var = 'n_unique_n_stop'\ntrain[var]=train['full_text'].apply(n_unique_words_no_stop)\ntest[var]=test['full_text'].apply(n_unique_words_no_stop)\n")


# In[18]:


plt.figure(figsize=(5,20))
sns.displot(data=train,x=var,kde="True")
plt.title('Number of Unique Words')
plt.show()


# ## Number of Non-words

# In[19]:


def n_non_words(text):
    text = text.translate(str.maketrans("", "", punctuation))
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    stop_words = ['a', 'i', ' ']
    #remove stop words
    words = [word for word in words if len(word) == 1]
    words = [word for word in words if word not in stop_words]
    unique_words = np.unique(words)
    return len(unique_words)


# In[20]:


get_ipython().run_cell_magic('time', '', "var = 'n_n_word'\ntrain[var]=train['full_text'].apply(n_non_words)\ntest[var]=test['full_text'].apply(n_non_words)\n")


# In[21]:


plt.figure(figsize=(5,20))
sns.displot(data=train,x=var,kde="True")
plt.title('Non-Word Count Distribution')
plt.show()


# # Number of Noun Phrases

# In[22]:


b = TextBlob("computer science artificial intelligence")
h = b.noun_phrases
len(h)


# In[23]:


def count_noun_phrases(text):
    blob = TextBlob(text)
    return len(blob.noun_phrases)


# In[24]:


get_ipython().run_cell_magic('time', '', "var = 'noun_phrase_count'\ntrain[var]=train['full_text'].apply(count_noun_phrases)\ntest[var]=test['full_text'].apply(count_noun_phrases)\n")


# In[25]:


plt.figure(figsize=(5,20))
sns.displot(data=train,x=var,kde="True")
plt.title('Number of Noun Phrases')
plt.show()


# # Parts of Speech

# https://www.nltk.org/book/ch05.html

# In[26]:


def pos(text):
    doc = nlp(text)
    result = dict(Counter([t.pos_ for t in doc]))
    missing = set(var) - set(result.keys())
    for miss in missing:
        result[miss] = np.nan
    return pd.Series(result)


# In[27]:


get_ipython().run_cell_magic('time', '', "var = ['PRON', 'VERB',\t'SCONJ', 'NOUN', 'AUX', 'ADP', 'PUNCT', 'PART',\t'CCONJ', 'ADV', 'DET', 'ADJ', 'SPACE', 'PROPN', 'NUM', 'INTJ', 'SYM', 'X']\ntrain[var]=train['full_text'].apply(pos)\ntest[var]=test['full_text'].apply(pos)\n")


# # Distributions

# In[28]:


plt.figure(figsize=(5,20))
sns.displot(data=train,x='PRON',kde="True")
plt.title('Number of Pronouns')
plt.show()


# In[29]:


plt.figure(figsize=(5,20))
sns.displot(data=train,x='VERB',kde="True")
plt.title('Number of Verbs')
plt.show()


# In[30]:


train = train.fillna(0.0)
test = test.fillna(0.0)


# # Sentiment Polarity

# In[31]:


def essay_polarity(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


# In[32]:


get_ipython().run_cell_magic('time', '', "var = 'polarity'\ntrain[var]=train['full_text'].apply(essay_polarity)\ntest[var]=test['full_text'].apply(essay_polarity)\n")


# # Essay Subjectivity

# In[33]:


def essay_subjectivity(text):
    blob = TextBlob(text)
    return blob.sentiment.subjectivity


# In[34]:


get_ipython().run_cell_magic('time', '', "var = 'subjectivity'\ntrain[var]=train['full_text'].apply(essay_subjectivity)\ntest[var]=test['full_text'].apply(essay_subjectivity)\n")


# # Grammar 
# 
# Potentially randomly sample sentences to get an idea of what the similarity is.

# In[35]:


sent = nlp("my name is nt the one you want")
a = TextBlob("my name is nt the one you want")
b = nlp(a.correct().string)
b


# In[36]:


sent.similarity(b)


# In[37]:


def spell_similarity(text): 
    b = nlp(TextBlob(text).correct().string)
    return nlp(text).similarity(b)


# In[38]:


get_ipython().run_cell_magic('time', '', "train['spell_score']=train['full_text'].apply(spell_similarity)\ntest['spell_score']=test['full_text'].apply(spell_similarity)\n")


# # Sentence Statistics

# In[39]:


def sentence_av_len_calc(text):
    sentences = pd.Series(text.split("."))
    sent_len = sentences.apply(n_unique_words)
    return np.mean(list(sent_len))

train['av_sent_len']=train['full_text'].apply(sentence_av_len_calc)
test['av_sent_len']=test['full_text'].apply(sentence_av_len_calc)


# In[40]:


def sentence_max_len_calc(text):
    sentences = pd.Series(text.split("."))
    sent_len = sentences.apply(n_unique_words)
    return np.max(list(sent_len))

train['max_sent_len']=train['full_text'].apply(sentence_max_len_calc)
test['max_sent_len']=test['full_text'].apply(sentence_max_len_calc)


# In[41]:


def sentence_min_len_calc(text):
    sentences = pd.Series(text.split("."))
    sent_len = sentences.apply(n_unique_words)
    return np.min(list(sent_len))

train['min_sent_len']=train['full_text'].apply(sentence_min_len_calc)
test['min_sent_len']=test['full_text'].apply(sentence_min_len_calc)


# In[42]:


def sentence_median_len_calc(text):
    sentences = pd.Series(text.split("."))
    sent_len = sentences.apply(n_unique_words)
    return np.median(list(sent_len))

train['med_sent_len']=train['full_text'].apply(sentence_median_len_calc)
test['med_sent_len']=test['full_text'].apply(sentence_median_len_calc)


# In[43]:


def sentence_std_len_calc(text):
    sentences = pd.Series(text.split("."))
    sent_len = sentences.apply(n_unique_words)
    return np.std(list(sent_len))

train['std_sent_len']=train['full_text'].apply(sentence_std_len_calc)
test['std_sent_len']=test['full_text'].apply(sentence_std_len_calc)


# # Number of sentences

# In[44]:


def sentence_count(text):
    sentences = text.split(".")
    return len(sentences)


# In[45]:


train['num_sent']=train['full_text'].apply(sentence_count)
test['num_sent']=test['full_text'].apply(sentence_count)


# In[46]:


plt.figure(figsize=(5,20))
sns.displot(data=train,x='num_sent',kde="True")
plt.title('Number of Sentences')
plt.show()


# # Sentiment Scores

# In[47]:


def generate_sentiment_scores(data):
    sid = SentimentIntensityAnalyzer()
    neg=[]
    pos=[]
    neu=[]
    comp=[]
    for sentence in tqdm(data['full_text'].values): 
        sentence_sentiment_score = sid.polarity_scores(sentence)
        comp.append(sentence_sentiment_score['compound'])
        neg.append(sentence_sentiment_score['neg'])
        pos.append(sentence_sentiment_score['pos'])
        neu.append(sentence_sentiment_score['neu'])
    return comp,neg,pos,neu
train['compound'],train['negative'],train['positive'],train['neutral']=generate_sentiment_scores(train)
test['compound'],test['negative'],test['positive'],test['neutral']=generate_sentiment_scores(test)


# # Character Length

# In[48]:


train['char_len']=train['full_text'].apply(lambda x:len(x.split()))
test['char_len']=test['full_text'].apply(lambda x:len(x.split()))


# In[49]:


y_train=train[['cohesion','syntax','vocabulary','phraseology','grammar','conventions']]


# In[50]:


train.to_csv('train_vars.csv', index=False)
y_train.to_csv('y_vars.csv', index=False)

