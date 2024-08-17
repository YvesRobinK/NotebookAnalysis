#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from termcolor import colored
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import bq_helper
from bq_helper import BigQueryHelper

import warnings
warnings.filterwarnings("ignore")


# In this competition, you will train your models on a novel semantic similarity dataset to extract relevant information by matching key phrases in patent documents. Determining the semantic similarity between phrases is critically important during the patent search and examination process to determine if an invention has been described before. For example, if one invention claims "television set" and a prior publication describes "TV set", a model would ideally recognize these are the same and assist a patent attorney or examiner in retrieving relevant documents.

# ### EVALUATION METRIC
# 
# **Pearson Correlation** is the coefficient that measures the degree of relationship between two random variables. The coefficient value ranges between +1 to -1. Pearson correlation is the normalization of covariance by the standard deviation of each random variable.
# 
# $$
# P C C(X, Y)=\frac{C O V(X, Y)}{S D_{x} * S D_{y}}
# $$
# ```
# X, Y: Two random variables
# COV(): covariance
# SD: standard deviation
# ```
# About Covariance:
# $$
# \operatorname{COV}(X, Y)=\frac{1}{n} * \sum_{i=1}^{n}\left(\left(X_{i}-\bar{X}\right) *\left(Y_{i}-\bar{Y}\right)\right)
# $$
# ```
# X, Y: Two random variables
# X_bar: mean of random variable X
# Y_bar: mean of random variable Y
# n: length of random variable X, Y
# ```
# About standard deviation:
# $$
# S D_{x}=\sqrt{\frac{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}{n}}
# $$
# ```
# X: random variables
# X_bar: mean of random variable X
# n: length of random variable X
# ```
# 
# 

# The host provided two files - train and test dataset.

# In[2]:


train_df = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv")
test_df = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/test.csv")


# In[3]:


print(f"Number of observations in TRAIN: {colored(train_df.shape, 'yellow')}")
print(f"Number of observations in TEST: {colored(test_df.shape, 'yellow')}")


# Let's look into first 20 observations in train dataset.

# In[4]:


train_df.sample(10)


# In this dataset, you are presented pairs of phrases (an **anchor and a target phrase**) and asked to rate how similar they are on a scale from 0 (not at all similar) to 1 (identical in meaning). This challenge differs from a standard semantic similarity task in that similarity has been scored here within a patent's context, specifically its CPC classification (version 2021.05), which indicates the subject to which the patent relates. For example, while the phrases "bird" and "Cape Cod" may have low semantic similarity in normal language, the likeness of their meaning is much closer if considered in the context of "house".
# 
# This is a code competition, in which you will submit code that will be run against an unseen test set. The unseen test set contains approximately 12k pairs of phrases. A small public test set has been provided for testing purposes, but is not used in scoring.

# In[5]:


test_df.sample(10)


# In[6]:


train_df.info()


# In[7]:


train_df.isnull().sum(axis = 0)


# Observations:
# - There is no empty rows

# In[8]:


train_df[train_df.drop("id", axis = 1).duplicated()]


# Observations:
# - There is no duplicates in train dataset

# ### ANCHOR COLUMN -  the first phrase

# In[9]:


print(f"Number of uniques values in ANCHOR column: {colored(train_df.anchor.nunique(), 'yellow')}")


# In[10]:


# TOP 20 anchors values
train_df.anchor.value_counts().head(20)


# In[11]:


pattern = 'base'
mask = train_df['target'].str.contains(pattern, case=False, na=False)
train_df.query("anchor =='component composite coating'")[mask]


# Observations:
# * we can see that there is "base coat", "basecoat" ... "layer basecoat", "layer basecoat coat" and "coating"
# * then "coat layer basecoat" ranked 0.5 and "coat layer basecoat coat" ranked 0.25

# In[12]:


anchor_desc = train_df[train_df.anchor.notnull()].anchor.values
stopwords = set(STOPWORDS) 
wordcloud = WordCloud(width = 800, 
                      height = 800,
                      background_color ='white',
                      min_font_size = 10,
                      stopwords = stopwords,).generate(' '.join(anchor_desc)) 

# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

plt.show()


# In[13]:


train_df['anchor_len'] = train_df['anchor'].str.split().str.len()

print(f"Anchors with maximum lenght of 5: \n{colored(train_df.query('anchor_len == 5')['anchor'].unique(), 'yellow')}")
print(f"\nAnchors with maximum lenght of 4: \n{colored(train_df.query('anchor_len == 4')['anchor'].unique(), 'green')}")


# In[14]:


train_df.anchor_len.hist(orientation='horizontal', color='#FFCF56')


# Observations:
# - Anchors are maximum 5 words in lenght 

# In[15]:


pattern = '[0-9]'
mask = train_df['anchor'].str.contains(pattern, na=False)
train_df['num_anchor'] = mask
train_df[mask]['anchor'].value_counts()


# Observations:
# * There are only 4 values containing numbers in train dataset

# ### TARGET COLUMN -  the second phrase

# In[16]:


print(f"Number of uniques values in TARGET column: {colored(train_df.target.nunique(), 'yellow')}")


# In[17]:


train_df.target.value_counts().head(20)


# In[18]:


target_desc = train_df[train_df.target.notnull()].target.values
stopwords = set(STOPWORDS) 
wordcloud = WordCloud(width = 800, 
                      height = 800,
                      background_color ='white',
                      min_font_size = 10,
                      stopwords = stopwords,).generate(' '.join(target_desc)) 

# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

plt.show() 


# In[19]:


train_df['target_len'] = train_df['target'].str.split().str.len()
train_df.target_len.value_counts()


# Observations:
# - Target are maximum 11 "words" in lenght 

# In[20]:


print(f"Targets with maximum lenght of 11: \n{colored(train_df.query('target_len == 11')['target'].unique(), 'yellow')}")
print(f"\nTargets with lenght of 10: \n{colored(train_df.query('target_len == 10')['target'].unique(), 'green')}")
print(f"\nTargets with lenght of 9: \n{colored(train_df.query('target_len == 9')['target'].unique(), 'yellow')}")
print(f"\nTargets with lenght of 8: \n{colored(train_df.query('target_len == 8')['target'].unique(), 'green')}")


# In[21]:


# Checking numbers in target feature

pattern = '[0-9]'
mask = train_df['target'].str.contains(pattern, na=False)
train_df['num_target'] = mask
train_df[mask]['target'].value_counts()


# Observations:
# * Target contains numbers and symbols
# * There are 112 observations where target feature contains numbers
# 

# In[22]:


pattern = '1 multiplexer'
mask = train_df['target'].str.contains(pattern, na=False)
train_df[mask]


# ### CONTEXT COLUMN

# Source: https://en.wikipedia.org/wiki/Cooperative_Patent_Classification
# 
# The first letter is the "section symbol" consisting of a letter from "A" ("Human Necessities") to "H" ("Electricity") or "Y" for emerging cross-sectional technologies. This is followed by a two-digit number to give a "class symbol" ("A01" represents "Agriculture; forestry; animal husbandry; trapping; fishing"). 
# 
# * A: Human Necessities
# * B: Operations and Transport
# * C: Chemistry and Metallurgy
# * D: Textiles
# * E: Fixed Constructions
# * F: Mechanical Engineering
# * G: Physics
# * H: Electricity
# * Y: Emerging Cross-Sectional Technologies

# * Hierarchy
#     * Section (one letter A to H and also Y)
#         * Class (two digits)
#         
# <div align="center"><img src="https://www.researchgate.net/publication/348420976/figure/fig2/AS:979346684645380@1610505853859/Example-of-a-simplified-Cooperative-Patent-Classification-CPC-tree-of-a-patent-parsed.ppm"/></div>

# In[23]:


print(f"Number of uniques values in CONTEXT column: {colored(train_df.context.nunique(), 'yellow')}")


# In[24]:


train_df.context.value_counts().head(20)


# We can create separate columns for **Section** and **Class**

# In[25]:


train_df['section'] = train_df['context'].astype(str).str[0]
train_df['classes'] = train_df['context'].astype(str).str[1:]
train_df.head(10)


# In[26]:


print(f"Number of uniques SECTIONS: {colored(train_df.section.nunique(), 'yellow')}")
print(f"Number of uniques CLASS: {colored(train_df.classes.nunique(), 'yellow')}")


# In[27]:


di = {"A" : "A - Human Necessities", 
      "B" : "B - Operations and Transport",
      "C" : "C - Chemistry and Metallurgy",
      "D" : "D - Textiles",
      "E" : "E - Fixed Constructions",
      "F" : "F- Mechanical Engineering",
      "G" : "G - Physics",
      "H" : "H - Electricity",
      "Y" : "Y - Emerging Cross-Sectional Technologies"}


# In[28]:


train_df.replace({"section": di}).section.hist(orientation='horizontal', color='#FFCF56')


# In[29]:


train_df.classes.value_counts().head(15)


# Addidtional datasets I found on Kaggle:
# * Cooperative Patent Classification (CPC) Data -> https://www.kaggle.com/datasets/bigquery/cpc
# * Cooperative Patent Classification Codes Meaning -> https://www.kaggle.com/datasets/xhlulu/cpc-codes

# In[30]:


# Cooperative Patent Classification (CPC) Data -> https://www.kaggle.com/datasets/bigquery/cpc

cpc = bq_helper.BigQueryHelper(active_project="cpc", dataset_name="cpc")

def get_cpc_row(cpc_code):
    query = f"""
    SELECT * FROM `patents-public-data.cpc.definition` WHERE symbol="{cpc_code}";
    """
    response = cpc.query_to_pandas_safe(query)
    return response

get_cpc_row('A47')


# In[31]:


# Cooperative Patent Classification Codes Meaning -> https://www.kaggle.com/datasets/xhlulu/cpc-codes
    
cpc_codes_df = pd.read_csv("../input/cpc-codes/titles.csv", dtype=str)
cpc_codes_df.head(10)


# In[32]:


cpc_codes_df.query("code == 'A47'")


# In[33]:


pd.options.display.max_colwidth
pd.options.display.max_colwidth = 100
cpc_codes_df.query("code == 'A47'").title


# In[34]:


# Let's join two datasets and add descriprion of context to our training DS

train_df['context_desc'] = train_df['context'].map(cpc_codes_df.set_index('code')['title']).str.lower()


# In[35]:


train_df.to_csv("us-train.csv", index = False)
train_df.sample(10)


# ### SCORE COLUMN
# 
# Score meanings
# The scores are in the 0-1 range with increments of 0.25 with the following meanings:
# 
# * 1.0 - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. “the”, “and”, “or”).
# * 0.75 - Close synonym, e.g. “mobile phone” vs. “cellphone”. This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
# * 0.5 - Synonyms which don’t have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
# * 0.25 - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
# * 0.0 - Unrelated.

# In[36]:


train_df.score.hist(color='#FFCF56')
train_df.score.value_counts()


# Look into very close match - score == 1

# In[37]:


train_df[['anchor', 'target', 'section', 'classes', 'score']].replace({"section": di}).query('score==1.0')


# Look into not related sentences - score == 0

# In[38]:


train_df[['anchor', 'target', 'section', 'classes', 'score']].replace({"section": di}).query('score==0.0')


# ### SUBMISSION FILE

# In[39]:


sub = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/sample_submission.csv")
sub.head(10)

