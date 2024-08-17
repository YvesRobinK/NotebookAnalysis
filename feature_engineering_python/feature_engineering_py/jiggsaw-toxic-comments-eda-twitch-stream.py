#!/usr/bin/env python
# coding: utf-8

# # Jigsaw/Conversation AI
# ## Toxic Comment Severity
# 
# ![toxic](https://miro.medium.com/max/2400/1*j6Ys0UcwbXIFoOnm5Zydkg.png)
# 
# EDA notebook, exploring the data provided for the Jigaw toxic Comment Severity competition.
# 
# In this competition we are given data from the **Wikipedia Talk page comments** dataset - and are asked to rank comments in order of toxicity.
# 
# The evaluation metric is **Average Agreement with Annotators (AAA?)** Where we must match *ranking* of the comment with that of annotators.
# 
# # Follow my Twitch live coding Streams...
# This notebook was created during a live coding stream on twitch. You can watch the video and follow for future videos here: https://www.twitch.tv/medallionstallion_
# During these streams I enjoy interacting with viewers, come and ask questions.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from itertools import cycle
plt.style.use('ggplot')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


# ## Lets take a look at the data.
# We are provided 3 csv files:
# - `validation_data.csv` - This contains pairs of rankings not from comments_to_score. It gives us an idea of how the rankings were applied. We also can learn about the annotators from this dataset.
# - `comments_to_score.csv` (aka test set)- for each comment text in this file, we need to rank these in order of toxicity.
# - `sample_submission.csv` - a sample submission file.
# 

# In[2]:


# Look at the data names and size
get_ipython().system('ls -Flash --color ../input/jigsaw-toxic-severity-rating/')


# In[3]:


val = pd.read_csv('../input/jigsaw-toxic-severity-rating/validation_data.csv')
comments = pd.read_csv('../input/jigsaw-toxic-severity-rating/comments_to_score.csv')
ss = pd.read_csv('../input/jigsaw-toxic-severity-rating/sample_submission.csv')
print(f'Validation Data csv is of shape: {val.shape}')
print(f'Comments csv is of shape: {comments.shape}')
print(f'Sample submission csv is of shape: {ss.shape}')


# ## Validation Data
# In this dataset we have three columns. The worker identifier - which is unique for the person ordering the pair of comments. Two columns `less_toxic` and `more_toxic` show the comments as the worker has ordered them.

# ## Comments most and lest commonly ranked `less_toxic` and `more_toxic`

# In[4]:


# Top 5 "Less Toxic" Comments.
val['less_toxic'].value_counts() \
    .to_frame().head(5)


# In[5]:


# Top 5 "More Toxic" Comments.
val['more_toxic'].value_counts() \
    .to_frame().head(5)


# ## Comment occurance in the validation set.
# How often to comments even appear in the validation set? What is the distribution, and what are the top/least occuring comments?
# 
# Some thing to note:
# 1. Comments tend to occur in multiples of 3 (3, 6, 9, etc.)
# 2. Most workers only score a small ammount of comments. However there are workers who score much more than the rest of the population (200+ pairs)

# In[6]:


all_comments = pd.concat([val['less_toxic'],
                          val['more_toxic']]) \
    .reset_index(drop=True)

ax = pd.DataFrame(index=range(1,19)) \
    .merge(all_comments.value_counts() \
           .value_counts().to_frame(),
           left_index=True, right_index=True, how='outer').fillna(0) \
    .astype('int').rename(columns={0:'Comment Frequency'}) \
    .plot(kind='bar',
          figsize=(12, 5))
plt.xticks(rotation=0)
ax.set_title('Comment Frequency in Val Dataset', fontsize=20)
ax.set_xlabel('Comment Occurance')
ax.set_ylabel('Number of Comments')
ax.legend().remove()
plt.show()


# In[7]:


ax = val['worker'].value_counts() \
    .plot(kind='hist', bins=50,
          color=color_pal[1], figsize=(12, 5))
ax.set_title('Frequeny of Worker in Val Set', fontsize=20)
ax.set_xlabel('Rows in Validation set for a Worker')


# In[8]:


# The most commonly occuring comment.
all_comments.value_counts() \
    .to_frame().rename(columns={0:'Total Comment Count'}) \
    .head()


# In[9]:


# The least common comment.
all_comments.value_counts() \
    .to_frame().rename(columns={0:'Total Comment Count'}) \
    .tail()


# ## Repeated Pairs in Validation Set
# How much workers agree and/or disagree.
# 1. Comment pairs occur in the same order 1, 2 or 3 times - but never more.
# 2. When we take the comments and undo the ordering (sort them alphabetically - we find that the pairs **almost always** occur 3 times)

# In[10]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
val['comment_pair_ordered'] = val['less_toxic'] + ' : ' + val['more_toxic']
# The most common pair
val['comment_pair_ordered'] \
    .value_counts().value_counts() \
    .plot(kind='bar', title='Ordered Comment Pairs',
          color=color_pal[4], ax=ax1)
ax1.tick_params(axis='x', rotation=0)
ax1.set_ylabel('Occurance')
ax1.set_xlabel('Number of times Pair is Found in Dataset')


# Comment Pairs in a standard alphabetical order
val['comment_pair_not_ordered'] = val[['less_toxic','more_toxic']] \
    .apply(lambda x: ':'.join(np.sort(list(x))), axis=1)
val['comment_pair_not_ordered'].value_counts().value_counts() \
    .sort_index() \
    .plot(kind='bar', title='Unordered Comment Pairs', ax=ax2,
          color=color_pal[5])
ax2.tick_params(axis='x', rotation=0)
ax2.set_xlabel('Number of times Unordered Pair is Found in Dataset')
plt.show()


# # Comments to Grade
# - Do they appear in the validation data? Yes 100% of the **public** `all_comments` also appear in the validation data. The private data may be a different story.

# In[11]:


comments['text'].isin(all_comments).mean()


# # Where do labelers disagree the most?
# We now know that pairs occur three times in the validation dataset. This leads us to ask the question... are there any "workers" who disagree more than others?
# 
# - We can create a new columns `n_agreements` to see for each row how many times the three workers had the same order for the given pair.

# In[12]:


val_order_dict = val['comment_pair_ordered'].value_counts().to_dict()
val['n_agreements'] = val['comment_pair_ordered'].map(val_order_dict)


# In[13]:


val['agreement'] = val['n_agreements'].map({1: 'Reviewer Disagreed',
                         2: 'Agreed with One Reviwer',
                         3: 'All Three Reviewers Agreed'})
ax = val['agreement'].value_counts().plot(kind='bar', color=color_pal[5],
                                         figsize=(12, 5))
ax.tick_params(axis='x', rotation=0)
ax.set_title('Worker Agreement', fontsize=16)
plt.show()


# In[14]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
# Reviewers with the most disagreements
val.query('n_agreements == 1')['worker'].value_counts(ascending=True) \
    .tail(20) \
    .plot(kind='barh', title='Reviewers with the Most Disagreements', ax=ax1)

# Reviewers with the most disagreements
val.query('n_agreements == 3')['worker'].value_counts(ascending=True) \
    .tail(20) \
    .plot(kind='barh', title='Reviewers with the Most Agreements', ax=ax2,
         color=color_pal[1])
plt.show()


# ## Lets look at disagreement count vs. total label reviews

# In[15]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

val['worker'].value_counts().to_frame().merge(
    val.query('n_agreements == 1')['worker'].value_counts().to_frame(),
    left_index=True, right_index=True
).rename(columns={'worker_x':'Number of Reviews',
                  'worker_y':'Number of Disagreements'}) \
    .plot(x='Number of Reviews', y='Number of Disagreements',
          kind='scatter', title='Worker Reviews vs Disagreements', ax=ax1)

val['worker'].value_counts().to_frame().merge(
    val.query('n_agreements == 3')['worker'].value_counts().to_frame(),
    left_index=True, right_index=True
).rename(columns={'worker_x':'Number of Reviews',
                  'worker_y':'Number of Disagreements'}) \
    .plot(x='Number of Reviews', y='Number of Disagreements',
          kind='scatter', title='Worker Reviews vs Agreements', ax=ax2, color=color_pal[2])
plt.show()


# # Wordclouds of Toxic and Non-Toxic Comments.

# In[16]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

non_toxic_comments = val['less_toxic'].value_counts() \
    .to_frame().head(1000)
non_toxic_text = ' '.join(non_toxic_comments.index.tolist())

toxic_comments = val['more_toxic'].value_counts() \
    .to_frame().head(1000)
toxic_text = ' '.join(toxic_comments.index.tolist())


wordcloud = WordCloud(max_font_size=50, max_words=100,width=500, height=500,
                      background_color="white") \
    .generate(non_toxic_text)


wordcloud2 = WordCloud(max_font_size=50, max_words=100,width=500, height=500,
                      background_color="black") \
    .generate(toxic_text)


fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,15))

ax1.imshow(wordcloud, interpolation="bilinear")
ax1.axis("off")
ax2.imshow(wordcloud2, interpolation="bilinear")
ax2.axis("off")
ax1.set_title('Non Toxic Comments', fontsize=25)
ax2.set_title('Toxic Comments', fontsize=25)
plt.show()


# # Simple Baseline Model using TFIDF and Linear Regression
# - We use a cleaned version of the dataset from the first jigsaw competition.
# - First we scrub the text dataset using some helper code
# - Then we convert text into vectorized representation using tfidf and bag of words.
# - We train logistic regression and linear regression on the `toxicity` feature.

# In[17]:


# A cleaned dataset from the first jigsaw competition
tox = pd.read_csv('../input/cleaned-toxic-comments/train_preprocessed.csv')

# Reference: https://www.kaggle.com/prateekarma/logistic-regression-with-feature-engineering
# https://stackoverflow.com/a/47091490/4084039
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])
pp_comments_to_score = []

for sentance in comments.text:
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    pp_comments_to_score.append(sentance.strip())


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import svm
from scipy.sparse import hstack


# Setup a config
cfg = {
    'MIN_DF_TFIDF' : 15,
    'MAX_FEATURES_TFIDF' : None,
    'MIN_DF_COUNT': 15,
    'MIN_FEATURES_COUNT': None,
}

# Create combined comments
combined_comments = pp_comments_to_score + tox.comment_text.tolist()
# Encode the text with the TFIDF vectorizor
vectorizer = TfidfVectorizer(min_df=cfg['MIN_DF_TFIDF'],
                             max_features=cfg['MAX_FEATURES_TFIDF'])
vectorizer.fit_transform(combined_comments)

train_comments_tfidf = vectorizer.transform(tox.comment_text)
comments_to_score_tfidf = vectorizer.transform(pp_comments_to_score)
print("Shape of training after tfidf ", train_comments_tfidf.shape)
print("Shape of test after tfidf", comments_to_score_tfidf.shape)


vectorizer = CountVectorizer(min_df=cfg['MIN_DF_COUNT'],
                             max_features=cfg['MIN_FEATURES_COUNT'])
vectorizer.fit(combined_comments)
train_comments_bow = vectorizer.transform(tox.comment_text)
comments_bow = vectorizer.transform(pp_comments_to_score)

feature_names_comments_bow_one_hot = vectorizer.get_feature_names()
print("Shape of matrix after bag of words",train_comments_bow.shape)
print("Shape of matrix after bag of words",comments_bow.shape)

x_train_stack = hstack([
    train_comments_tfidf,
    train_comments_bow
])

x_test_stack = hstack([
    comments_to_score_tfidf, 
    comments_bow
])


# ## Linear Regression Model

# In[19]:


# Linear Regression
ss_lr = ss.copy()
lr = LinearRegression()
lr.fit(x_train_stack, tox['toxicity'].values)
ss_lr['score'] = lr.predict(x_test_stack)
ss_lr['score'] = ss_lr['score'].rank(method='first')
ss_lr.to_csv('submission.csv', index=False)


# ## Logistic Regression Model

# In[20]:


# Logistic Regression Model
ss_logr = ss.copy()
logr = LogisticRegression(solver='liblinear')
logr.fit(x_train_stack, tox['toxicity'].values)
ss_logr['score'] = logr.predict(x_test_stack)
ss_logr['score'] = ss_logr['score'].rank(method='first')
ss_logr.to_csv('submission-logistic.csv', index=False)


# ## Blend

# In[21]:


ss_blend = ss.copy()
ss_blend['score_lr'] = ss_lr['score']
ss_blend['score_logr'] = ss_logr['score']
ss_blend['score'] = ss_blend[['score_lr','score_logr']].mean(axis=1)
ss_blend['score'] = ss_blend['score'].rank(method='first')
ss_blend[['comment_id','score']].to_csv('submission-blend.csv', index=False)


# # Compare Model Predictions
# - Both models are able to pull the REALLY Toxic comments to the top!

# In[22]:


ss_blend.plot(kind='scatter',
              x='score_lr', y='score_logr',
              figsize=(12, 12),
              title='Logistic vs Linear Model Predictions')
plt.show()

