#!/usr/bin/env python
# coding: utf-8

# # Google QUEST Q&A Labeling
# 
# **Improving automated understanding of complex question answer content**
# 
# > Computers are really good at answering questions with single, verifiable answers. But, humans are often still better at answering questions about opinions, recommendations, or personal experiences.
# > ...
# > In this competition, you’re challenged to use this new dataset to build predictive algorithms for different subjective aspects of question-answering.
# 
# ![](https://storage.googleapis.com/kaggle-media/competitions/google-research/human_computable_dimensions_1.png)
# 
# 
# The competition is **Notebook-only competition**. Your Notebook will re-run automatically against an unseen test set.
# 
# This competition data is small, only made of 6079 rows of train dataset.<br/>
# So I think this competition is **easy for beginners to participate** in terms of computational resource (unless you use BERT or any other heavy models to get good score), compared to the past competition hosted by Google like Open Image Challenges which requires a lot of GPU resources to train the model.

# In[1]:


import gc
import os
from pathlib import Path
import random
import sys

from tqdm import tqdm_notebook as tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


# In[2]:


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


# # Data loading and data explanation

# In[3]:


get_ipython().run_cell_magic('time', '', "datadir = Path('/kaggle/input/google-quest-challenge')\n\n# Read in the data CSV files\ntrain = pd.read_csv(datadir/'train.csv')\ntest = pd.read_csv(datadir/'test.csv')\nsample_submission = pd.read_csv(datadir/'sample_submission.csv')\n")


# Let's check each data size.
# 
# Train and test data consists of 6079 rows and 476 rows respectively.<br/>
# We have 30 different target labels to predict.<br/>
# Rest 10 columns are given as feature.

# In[4]:


print('train', train.shape)
print('test', test.shape)
print('sample_submission', sample_submission.shape)


# ## target labels
# 
# Let's check target labels at first.
# 
# Each row is identified by question id: `qa_id`, and other 30 columns are target labels.

# In[5]:


sample_submission.head()


# In[6]:


sample_submission.columns


# 30 target labels consist of 21 question related labels and 9 answer related labels.
# 
# NOTE: the labels are given in the continuous range from [0, 1]. NOT binary value.
# 
# > This is not a binary prediction challenge. Target labels are aggregated from multiple raters, and can have continuous values in the range [0,1]. Therefore, predictions must also be in that range.

# ## feature columns
# 
# Let's check feature columns one by one.

# In[7]:


feature_columns = [col for col in train.columns if col not in sample_submission.columns]
print('Feature columns: ', feature_columns)


# Each row contains question and answer information together with the original Q&A page URL of the StackExchange properties.

# In[8]:


train[feature_columns].head()


# Let's focus on the first row of the data. You can access original page mentioned in the `url` column.
# 
#  - [https://photo.stackexchange.com/questions/9169/what-am-i-losing-when-using-extension-tubes-instead-of-a-macro-lens](https://photo.stackexchange.com/questions/9169/what-am-i-losing-when-using-extension-tubes-instead-of-a-macro-lens)
#  
#  
#  Only the question contains "title" (`question_title`), and we have `question_body` and `answer` which is given by sentences.

# In[9]:


train0 = train.iloc[0]

print('URL           : ', train0['url'])
print('question_title: ', train0['question_title'])
print('question_body : ', train0['question_body'])


# In[10]:


print('answer        : ', train0['answer'])


# When you access to the URL, you can understand that multiple answer to the single question is given in the page. But only one answer is sampled in the dataset.
# Also this answer may not be the most popular answer. We can find the answer of this data in the relatively bottom part of the homepage.

# Other columns are metadata, which shows **question user** property, **answer user** property and **category** of question.

# In[11]:


train[['url', 'question_user_name', 'question_user_page', 'answer_user_name', 'answer_user_page', 'url', 'category', 'host']]


# # Exploratory Data Analysis
# 
# Let's check each column of the data more carefully.

# ## target label distribution

# In[12]:


target_cols = ['question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written']


# In[13]:


train[target_cols]


# In[14]:


fig, axes = plt.subplots(6, 5, figsize=(18, 15))
axes = axes.ravel()
bins = np.linspace(0, 1, 20)

for i, col in enumerate(target_cols):
    ax = axes[i]
    sns.distplot(train[col], label=col, kde=False, bins=bins, ax=ax)
    # ax.set_title(col)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 6079])
plt.tight_layout()
plt.show()
plt.close()


# It seems some of the labels are quite imbalanced. For example "question_not_really_a_question" is almost always 0, which means most of the question in the data is not a noisy data but an "actual question".

# ## Nan values
# 
# There is no nan values in the data.

# In[15]:


train.isna().sum()


# In[16]:


test.isna().sum()


# ## Category
# 
# The dataset consists of 5 categories: "Technology", "Stackoverflow", "Culture", "Science", "Life arts".<br/>
# Train/Test distribution is almost same.

# In[17]:


train_category = train['category'].value_counts()
test_category = test['category'].value_counts()


# In[18]:


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
train_category.plot(kind='bar', ax=axes[0])
axes[0].set_title('Train')
test_category.plot(kind='bar', ax=axes[1])
axes[1].set_title('Test')
print('Train/Test category distribution')


# ## Word Cloud visualization
# 
# Let's see what kind of word are used for question and answer. Also let's check the difference between train and test.

# In[19]:


from wordcloud import WordCloud


def plot_wordcloud(text, ax, title=None):
    wordcloud = WordCloud(max_font_size=None, background_color='white',
                          width=1200, height=1000).generate(text_cat)
    ax.imshow(wordcloud)
    if title is not None:
        ax.set_title(title)
    ax.axis("off")


# In[20]:


print('Training data Word Cloud')

fig, axes = plt.subplots(1, 3, figsize=(16, 18))

text_cat = ' '.join(train['question_title'].values)
plot_wordcloud(text_cat, axes[0], 'Question title')

text_cat = ' '.join(train['question_body'].values)
plot_wordcloud(text_cat, axes[1], 'Question body')

text_cat = ' '.join(train['answer'].values)
plot_wordcloud(text_cat, axes[2], 'Answer')

plt.tight_layout()
fig.show()


# In[21]:


print('Test data Word Cloud')

fig, axes = plt.subplots(1, 3, figsize=(16, 18))

text_cat = ' '.join(test['question_title'].values)
plot_wordcloud(text_cat, axes[0], 'Question title')

text_cat = ' '.join(test['question_body'].values)
plot_wordcloud(text_cat, axes[1], 'Question body')

text_cat = ' '.join(test['answer'].values)
plot_wordcloud(text_cat, axes[2], 'Answer')

plt.tight_layout()
fig.show()


# It seems common word usage distribution is similar between train & test dataset!

# ## Correlation in target labels
# 
# I could find following 3 pairs are **correlated**:
# 
#  - "question_type_instructions" & "answer_type_instructions"
#  - "question_type_procedure" & "answer_type_procedure"
#  - "question_type_reason_explanation" & "answer_type_reason_explanation" 
# 
# This is reasonable that same evaluation on both question & answer are correlated.
# 
# On the other hand, **Anticorrelation** pattern can be found on following pairs:
# 
#  - "question_fact_seeking" & "question_opinion_seeking"
#  - "answer_type_instruction" & "answer_type_reason_explanation"
# 
# I think this is also reasonable that question that asks fact & opinion conflicts.<br/>
# And answer which shows instruction or reason explanation also conflicts.

# In[22]:


fig, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(train[target_cols].corr(), ax=ax)


# ## User check
# 
# The dataset contains question user and answer user information. This may be because user attribution is impotant, same user tend to answer same kind of question and same answer user tends to answer in similar quality.
# 
# Let's check if how the user are distributed, and the user are duplicated in train/test or not.

# In[23]:


train_question_user = train['question_user_name'].unique()
test_question_user = test['question_user_name'].unique()

print('Number of unique question user in train: ', len(train_question_user))
print('Number of unique question user in test : ', len(test_question_user))
print('Number of unique question user in both train & test : ', len(set(train_question_user) & set(test_question_user)))


# In[24]:


train_answer_user = train['answer_user_name'].unique()
test_answer_user = test['answer_user_name'].unique()

print('Number of unique answer user in train: ', len(train_answer_user))
print('Number of unique answer user in test : ', len(test_answer_user))
print('Number of unique answer user in both train & test : ', len(set(train_answer_user) & set(test_answer_user)))


# Seems several users are in both train & test dataset.
# 
# Also, it seems many users ask question and answer.

# In[25]:


print('Number of unique user in both question & anser in train  : ', len(set(train_answer_user) & set(train_question_user)))
print('Number of unique user in both question & anser in train  : ', len(set(test_answer_user) & set(test_question_user)))


# So these user information maybe important to predict `test` dataset!

# # Simple feature engineering
# 
# Now, I will proceed simple feature engineering and check if it explains data well or not.
# 
#  - Number of words in question title, body and answer.
#  - question_user's question count in train.
#  - answer_user's answer count in train.
#  
# Work in progress... Maybe I will write in another kernel...

# ## Number of words

# In[26]:


def char_count(s):
    return len(s)

def word_count(s):
    return s.count(' ')


# In[27]:


train['question_title_n_chars'] = train['question_title'].apply(char_count)
train['question_title_n_words'] = train['question_title'].apply(word_count)
train['question_body_n_chars'] = train['question_body'].apply(char_count)
train['question_body_n_words'] = train['question_body'].apply(word_count)
train['answer_n_chars'] = train['answer'].apply(char_count)
train['answer_n_words'] = train['answer'].apply(word_count)

test['question_title_n_chars'] = test['question_title'].apply(char_count)
test['question_title_n_words'] = test['question_title'].apply(word_count)
test['question_body_n_chars'] = test['question_body'].apply(char_count)
test['question_body_n_words'] = test['question_body'].apply(word_count)
test['answer_n_chars'] = test['answer'].apply(char_count)
test['answer_n_words'] = test['answer'].apply(word_count)


# **Number of chars and words in Question title**

# In[28]:


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.distplot(train['question_title_n_chars'], label='train', ax=axes[0])
sns.distplot(test['question_title_n_chars'], label='test', ax=axes[0])
axes[0].legend()
sns.distplot(train['question_title_n_words'], label='train', ax=axes[1])
sns.distplot(test['question_title_n_words'], label='test', ax=axes[1])
axes[1].legend()


# **Number of chars and words in Question body**

# In[29]:


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.distplot(train['question_body_n_chars'], label='train', ax=axes[0])
sns.distplot(test['question_body_n_chars'], label='test', ax=axes[0])
axes[0].legend()
sns.distplot(train['question_body_n_words'], label='train', ax=axes[1])
sns.distplot(test['question_body_n_words'], label='test', ax=axes[1])
axes[1].legend()


# Outlier has too long, let's cut these outlier for visualization.

# In[30]:


train['question_body_n_chars'].clip(0, 5000, inplace=True)
test['question_body_n_chars'].clip(0, 5000, inplace=True)
train['question_body_n_words'].clip(0, 1000, inplace=True)
test['question_body_n_words'].clip(0, 1000, inplace=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.distplot(train['question_body_n_chars'], label='train', ax=axes[0])
sns.distplot(test['question_body_n_chars'], label='test', ax=axes[0])
axes[0].legend()
sns.distplot(train['question_body_n_words'], label='train', ax=axes[1])
sns.distplot(test['question_body_n_words'], label='test', ax=axes[1])
axes[1].legend()


# **Number of chars and words in answer**
# 
# Answer number chars/words distribution is similar to question body.

# In[31]:


train['answer_n_chars'].clip(0, 5000, inplace=True)
test['answer_n_chars'].clip(0, 5000, inplace=True)
train['answer_n_words'].clip(0, 1000, inplace=True)
test['answer_n_words'].clip(0, 1000, inplace=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.distplot(train['answer_n_chars'], label='train', ax=axes[0])
sns.distplot(test['answer_n_chars'], label='test', ax=axes[0])
axes[0].legend()
sns.distplot(train['answer_n_words'], label='train', ax=axes[1])
sns.distplot(test['answer_n_words'], label='test', ax=axes[1])
axes[1].legend()


# Are these feature useful for predicting target values?<br/>
# Let's check correlation with target values.

# In[32]:


from scipy.spatial.distance import cdist

def calc_corr(df, x_cols, y_cols):
    arr1 = df[x_cols].T.values
    arr2 = df[y_cols].T.values
    corr_df = pd.DataFrame(1 - cdist(arr2, arr1, metric='correlation'), index=y_cols, columns=x_cols)
    return corr_df


# In[33]:


number_feature_cols = ['question_title_n_chars', 'question_title_n_words', 'question_body_n_chars', 'question_body_n_words', 'answer_n_chars', 'answer_n_words']
# train[number_feature_cols].corrwith(train[target_cols], axis=0)

corr_df = calc_corr(train, target_cols, number_feature_cols)


# In[34]:


corr_df


# In[35]:


fig, ax = plt.subplots(figsize=(25, 5))
sns.heatmap(corr_df, ax=ax)


# We can see following relationship
# 
#  - length of answer is correlated with "answer_level_of_information".
#  - length of question_title is correlated with "question_body_critical" and length of question body is anticorrelated with it.
#  - length of question_body is anticorrelated with "question_well_written"

# ## Number of question or answer by user

# In[36]:


num_question = train['question_user_name'].value_counts()
num_answer = train['answer_user_name'].value_counts()

train['num_answer_user'] = train['answer_user_name'].map(num_answer)
train['num_question_user'] = train['question_user_name'].map(num_question)
test['num_answer_user'] = test['answer_user_name'].map(num_answer)
test['num_question_user'] = test['question_user_name'].map(num_question)

# # map is done by train data, we need to fill value for user which does not appear in train data...
# test['num_answer_user'].fillna(1, inplace=True)
# test['num_question_user'].fillna(1, inplace=True)


# In[37]:


number_feature_cols = ['num_answer_user', 'num_question_user']
# train[number_feature_cols].corrwith(train[target_cols], axis=0)

corr_df = calc_corr(train, target_cols, number_feature_cols)


# In[38]:


fig, ax = plt.subplots(figsize=(30, 2))
sns.heatmap(corr_df, ax=ax)


# Although correlation scale is small and it might not be a "true correlation", I can see following pattern:
# 
#  - `num_question_user` and `question_conversational` is correlated: People who post question a lot tend to ask question in conversational form.

# In[ ]:





# That's all for the start introduction of this competition!
# 
# <h3 style="color:red">If this kernel helps you, please upvote to keep me motivated :)<br>Thanks!</h3>
