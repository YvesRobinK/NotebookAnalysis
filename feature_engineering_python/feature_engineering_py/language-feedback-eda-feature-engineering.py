#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os
import seaborn as sns
import matplotlib.pyplot as plt
import nltk # Tool for text analysis
from nltk import word_tokenize # For total number of words and modeling challenges
from wordcloud import WordCloud # Visualization of the most representative words
from pandas.api.types import CategoricalDtype # Later data treatment
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Dataset Information

# In[2]:


#Train dataset
train=pd.read_csv("/kaggle/input/feedback-prize-english-language-learning/train.csv")
train.head()


# In[3]:


test=pd.read_csv("/kaggle/input/feedback-prize-english-language-learning/test.csv")
test


# For additional insights, the describe function was used, identifying that the mean of all of the categories is close to three while the standard deviation is on an average of 0.65. At the same time, with the .info() property it was recognized that there are no null values in any of the columns.

# In[4]:


# Numerical Analysis
train.describe()


# In[5]:


#Dataframe composition
train.info()


# As an additional way of visualizing the data, the seaborn library was used to create the following histograms, showing that all the categories have a similar distribution (close to the normal) and only the vocabulary has a few outliers, representing that the highest and lowest scores there are few texts with those numbers. This implies a challenge in the future prediction.

# In[6]:


#Analysis of total representarion per category

train_scores=train.drop(columns=["full_text","text_id"])
n=1
plt.figure(figsize=(15,20))
for i in train_scores.columns:
    #Use of subplot for better presentation
    plt.subplot(3,2,n)
    plt.grid()
    sns.histplot(data=train_scores,x=i)
    n +=1


# ***

# # Text Analysis

# For an initial look at the database, the total length of each of the texts was calculated through a lambda function and the string function split. With those values, a histogram was plotted, showing that it first looks like a right-skewed distribution as of the effect of the outliers, as the mean is close to 500 words.

# In[7]:


total_words=train.full_text.apply(lambda x: len(x.split(" ")))
plt.figure(figsize=(10,10))
sns.histplot(total_words)
plt.grid()
plt.title("Distribution of total text Length");


# In[8]:


#Total mean
total_words.mean()


# Now, for looking at the general topic of the essays a word cloud was used with all the available words in the text. As expected, the most repeated words correspond to school topics such as student and school.

# In[9]:


word_cloud_text = ''.join(train.full_text)

wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="black",
    scale=10,
    width=800,
    height=400
).generate(word_cloud_text)

plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# With these data, the main text was transformed to lowercase, as a way to be able to compare the different types of words with the list of words from nltk. Then, a word tokenize comparison was made to show the difference of the total unique words making those changes, reducing by almost 3000 tokens.

# In[10]:


# Lowercase text with string method lower()
train['full_text_low'] = train.full_text.apply(lambda x: x.lower())


# In[11]:


#Creation of unique tokens
token_lists = [word_tokenize(each) for each in train.full_text]
tokens = [item for sublist in token_lists for item in sublist]
print("Number of unique tokens: ", len(set(tokens)))


# In[12]:


#Number of unique words with the lowercase transformation
token_lists_lower = [word_tokenize(each) for each in train.full_text_low]
tokens_lower = [item for sublist in token_lists_lower for item in sublist]
print("Number of unique tokens with lowercase: ", len(set(tokens_lower)))


# After this, from nltk, the stopwords from English were loaded. For example, they correspond to I, me, and other types of pronouns. With these data, the final plots were made, as a way to identify features that could affect the score in each of the categories.

# In[13]:


#Download of stopwords
nltk.download('stopwords')
stopwords_corpus = nltk.corpus.stopwords
eng_stop_words = stopwords_corpus.words('english')
print(len(eng_stop_words))
eng_stop_words[:10]


# ***

# # Feature Engineering and EDA

# Finally, two columns were added to the main dataframe, corresponding to the total length of each of the essays and the total number of stopwords.

# In[14]:


# Length of texts
train["Length"]=train.full_text.apply(lambda x: len(x.split(" ")))
# Total number of stopwords
train["Stopwords"]=train.full_text_low.apply(lambda x: len([w for w in x.split(" ") if w in eng_stop_words]))


# Additionally, the score columns were transformed to categorical as a way to build the final boxplots.

# In[15]:


# Category object
type_at=CategoricalDtype(ordered=True)
# Train dataframe transformation
analysis_columns=["cohesion","syntax","vocabulary","phraseology","grammar","conventions"]
for i in analysis_columns:
    train[i]=train[i].astype(type_at)
train.info()


# Then, the first group of plots corresponds to the relation between the length of the essays and the scores. As an initial insight, the mean length of each of the categories increases as the score is higher, however, there are a lot of outliers that could affect the development of future models.
# 
# At the same time, the next group of plots that represent the relation between the number of stopwords and the score shows a more evident relation between these values even when there are also a big number of outliers.

# In[16]:


# Relation between length and score
plt.figure(figsize=(15,20))
n=1
for i in analysis_columns:
    plt.subplot(3,2,n)
    plt.grid()
    sns.boxplot(data=train,x="Length",y=i)
    n +=1


# In[17]:


# Relation between number of stopwords and score
plt.figure(figsize=(15,20))
n=1
for i in analysis_columns:
    plt.subplot(3,2,n)
    plt.grid()
    sns.boxplot(data=train,x="Stopwords",y=i)
    n +=1


# ***

# # Conclusions

# * The essay length seems to be an important factor that is related to the total score of each of the 6 categories, but with a high number of outliers.
# * The number of stopwords has a similar trend, that could be used for future modeling.
# * There is a great number of unique words (more than 20000) that need to be considered in the model construction.
# * The total values for each category follow a normal distribution, being an important fact for using statistical modeling.
# ***

# --------  Please do Upvote if you liked my work ------------

# In[ ]:




