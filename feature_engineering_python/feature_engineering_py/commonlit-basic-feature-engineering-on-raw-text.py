#!/usr/bin/env python
# coding: utf-8

# # What's In the Excerpt?
# 
# This notebook will explore various hypothesis I have on the problem, and in the go feature engineer and perform exploratory data analysis

# In[1]:


import numpy as np # linear algebr/a
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

from pandas.io.json import json_normalize
from plotly import tools
py.init_notebook_mode(connected=True)
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
color = sns.color_palette()
np.random.seed(13)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_df = pd.read_csv("../input/commonlitreadabilityprize/train.csv")
test_df = pd.read_csv("../input/commonlitreadabilityprize/test.csv")


# In[3]:


train_df


# ## Exploration of text value : excerpt

# Before we do anything lets begin by seeing few examples and trying to read it out loud. Yeah, I am doing a reading practise 😅

# In[4]:


sample_idx = [456, 784, 33]

for idx in sample_idx: 
    print("============================")
    print(f">> Sample example #{idx}")
    print("============================")
    print(train_df.iloc[idx]['excerpt'],"\n\n", f"=> Score {train_df.iloc[idx]['target']}\n\n")


# In[5]:


train_df['length'] = train_df['excerpt'].apply(len)


# In[6]:


from tqdm import tqdm # I love this handy tool! 
print(">> Generating Count Based And Demographical Features")
for df in ([train_df,test_df]):
    df['length'] = df['excerpt'].apply(lambda x : len(x))
    df['capitals'] = df['excerpt'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['length']),axis=1)
    df['num_exclamation_marks'] = df['excerpt'].apply(lambda comment: comment.count('!'))
    df['num_question_marks'] = df['excerpt'].apply(lambda comment: comment.count('?'))
    df['num_punctuation'] = df['excerpt'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
    df['num_symbols'] = df['excerpt'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
    df['num_words'] = df['excerpt'].apply(lambda comment: len(comment.split()))
    df['num_unique_words'] = df['excerpt'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
 


def tag_part_of_speech(text):
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    pos_list = pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    adjective_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    verb_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return[noun_count, adjective_count, verb_count]

print(">> Generating POS Features")
for df in ([train_df,test_df]):
    df['nouns'], df['adjectives'], df['verbs'] = zip(*df['excerpt'].apply(
        lambda comment: tag_part_of_speech(comment)))
    df['nouns_vs_length'] = df['nouns'] / df['length']
    df['adjectives_vs_length'] = df['adjectives'] / df['length']
    df['verbs_vs_length'] = df['verbs'] /df['length']
    df['nouns_vs_words'] = df['nouns'] / df['num_words']
    df['adjectives_vs_words'] = df['adjectives'] / df['num_words']
    df['verbs_vs_words'] = df['verbs'] / df['num_words']
    # More Handy Features
    df["count_words_title"] = df["excerpt"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df["mean_word_len"] = df["excerpt"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df['punct_percent']= df['num_punctuation']*100/df['num_words']


# In[7]:


train_df[['nouns','nouns_vs_length','adjectives_vs_length','verbs_vs_length','nouns_vs_words','adjectives_vs_words','verbs_vs_words']].head(8)


# In[8]:


f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(train_df.drop(['id','excerpt'], axis=1).corr(), annot=True, fmt=".2f", ax=ax, 
            cbar_kws={'label': 'Correlation Coefficient'}, cmap='viridis')
ax.set_title("Correlation Matrix for Target and New Features", fontsize=20)
plt.show()


# In[9]:


train_df.to_csv("commonlit_feat_train.csv", index=None)
test_df.to_csv("commonlit_feat_test.csv", index=None)


# In[ ]:




