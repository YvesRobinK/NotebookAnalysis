#!/usr/bin/env python
# coding: utf-8

# <h1><center>Jigsaw Toxic Comments Classification</center></h1>
# <h2><center>Simple EDA, Cleaning with Tensorflow Embedding Baseline!</center></h2>

# ![Toxic Comments](https://miro.medium.com/max/1400/1*8BdmU3wYefT7vDZRWWOL1Q.png)

# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:Purple; border:0; color:white' role="tab" aria-controls="home"><center>Contents</center></h2>

# 1. [Imports](#imports)  
# 2. [Load Datasets](#load-datasets)  
# 3. [Exploratory Data Analysis](#eda)  
# 4. [Handling Imbalanced Dataset](#handle-imbalanced-dataset) 
# 5. [Feature Engineering](#feature-engineering)   
# 6. [Text Preprocessing](#text-preprocessing)
# 7. [Model Definition](#model-definition)
# 8. [Model Training](#model-training)
# 9. [Prediction](#prediction)   
# 10. [References](#references) 

# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:magents; border:0; color:white' role="tab" aria-controls="home"><center>If you find this notebook useful, ***Do Upvote***. Feel free to share your feedback in comments</center></h2>

# <a id="imports"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:Purple; border:0; color:white' role="tab" aria-controls="home"><center>Imports</center></h2>

# In[2]:


import numpy as np
import pandas as pd
import string

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from wordcloud import STOPWORDS

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# <a id="load-datasets"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:Purple; border:0; color:white' role="tab" aria-controls="home"><center>Load Datasets</center></h2>

# In[3]:


train = pd.read_csv('../input/toxic-comments-train/train.csv')
test = pd.read_csv('../input/jigsaw-toxic-severity-rating/validation_data.csv')
sample = pd.read_csv('../input/jigsaw-toxic-severity-rating/sample_submission.csv')
target = pd.read_csv('../input/jigsaw-toxic-severity-rating/comments_to_score.csv')


# <a id="eda"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:Purple; border:0; color:white' role="tab" aria-controls="home"><center>Exploratory Data Analysis</center></h2>

# In[4]:


train.head()


# In[5]:


train['y'] = train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].sum(axis=1) > 0
train.drop(['toxic','severe_toxic','obscene','threat','insult','identity_hate'], inplace=True, axis=1)


# In[6]:


train.head()


# In[7]:


train.y.unique()


# In[8]:


train.y.value_counts().plot(kind='barh')


# <a id="handle-imbalanced-dataset"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:Purple; border:0; color:white' role="tab" aria-controls="home"><center>Handling Imbalanced Dataset</center></h2>

# #### We can clearly see we have imbalanced dataset. Let's fix it

# In[9]:


count_of_toxic_comments =  train[train.y != 0].shape[0]
count_of_toxic_comments


# In[10]:


train_toxic = train[train.y != 0]
train_non_toxic = train[train.y == 0].sample(count_of_toxic_comments)


# In[11]:


df = pd.concat([train_toxic, train_non_toxic])
df


# In[12]:


df.y.value_counts().plot(kind='barh')


# #### Imbalanced dataset issue sorted

# <a id="feature-engineering"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:Purple; border:0; color:white' role="tab" aria-controls="home"><center>Feature Engineering</center></h2>

# In[13]:


# word_count
df['word_count'] = df['comment_text'].apply(lambda x: len(str(x).split()))

# unique_word_count
df['unique_word_count'] = df['comment_text'].apply(lambda x: len(set(str(x).split())))

# stop_word_count
df['stop_word_count'] = df['comment_text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

# mean_word_length
df['mean_word_length'] = df['comment_text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# char_count
df['char_count'] = df['comment_text'].apply(lambda x: len(str(x)))

# punctuation_count
df['punctuation_count'] = df['comment_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))


# In[14]:


df.head()


# In[15]:


METAFEATURES = ['word_count', 'unique_word_count', 'stop_word_count', 'mean_word_length','char_count', 'punctuation_count']
TOXIC_COMMENTS = df['y'] == 1

fig, axes = plt.subplots(ncols=2, nrows=len(METAFEATURES), figsize=(20, 50), dpi=100)

for i, feature in enumerate(METAFEATURES):
    sns.distplot(df.loc[~TOXIC_COMMENTS][feature], label='Non Toxic', ax=axes[i][0], color='green')
    sns.distplot(df.loc[TOXIC_COMMENTS][feature], label='Toxic', ax=axes[i][0], color='red')

    sns.distplot(df[feature], label='Train', ax=axes[i][1])
    
    for j in range(2):
        axes[i][j].set_xlabel('')
        axes[i][j].tick_params(axis='x', labelsize=12)
        axes[i][j].tick_params(axis='y', labelsize=12)
        axes[i][j].legend()
    
    axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize=13)

plt.show()


# In[16]:


df.describe()


# <a id="text-preprocessing"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:Purple; border:0; color:white' role="tab" aria-controls="home"><center>Text Preprocessing</center></h2>

# ### 1. Remove stopwords, Punctuations

# In[17]:


# Remove stopwords & convert to lower case
df['comment_text'] = df['comment_text'].apply(lambda x: ' '.join([w for w in str(x).lower().split() if w not in STOPWORDS]))

# Remove Punctuations
df["comment_text"] = df['comment_text'].str.replace('[^\w\s]','')
df.tail()


# In[18]:


X = df.drop(['y'], axis=1)
y = df['y']


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[20]:


X_train.head()


# In[21]:


X_train = X_train.comment_text.values
X_test = X_test.comment_text.values


# In[22]:


y_train.head()


# #### Lets convert words to numbers

# In[23]:


OOV_TOKEN = '<OOV>'
VOCAB_SIZE = 10000
MAX_LEN = 100
EMBEDDING_DIM = 100


# ### 2. Tokenization

# In[24]:


tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token=OOV_TOKEN
)
tokenizer.fit_on_texts(X_train)


# In[25]:


len(tokenizer.word_index)


# ### 3. Convert text to padded sequences

# In[26]:


train_seq = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(
    train_seq, maxlen=MAX_LEN, dtype='int32', padding='post',
    truncating='post'
)

test_seq = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(
    test_seq, maxlen=MAX_LEN, dtype='int32', padding='post',
    truncating='post'
)


# In[27]:


test_padded.shape


# <a id="model-definition"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:Purple; border:0; color:white' role="tab" aria-controls="home"><center>Model Definition</center></h2>

# In[29]:


model = tf.keras.Sequential([
  Embedding(VOCAB_SIZE, EMBEDDING_DIM, name="embedding"),
    LSTM(64),
    Dropout(0.2),
  Dense(16, activation='relu'),
    Dropout(0.2),
  Dense(1,activation='sigmoid')
])


# In[30]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[31]:


model.summary()


# <a id="model-training"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:Purple; border:0; color:white' role="tab" aria-controls="home"><center>Model Training</center></h2>

# In[32]:


es = EarlyStopping(patience=3, 
                   monitor='loss', 
                   restore_best_weights=True, 
                   mode='min', 
                   verbose=1)


# In[42]:


hist = model.fit(
    train_padded,
    y = y_train,
    validation_data=(test_padded, y_test),
    epochs=15,
    callbacks=es
)


# In[43]:


plt.style.use('fivethirtyeight')

# visualize the models accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc = 'upper left')
plt.show()  


# <a id="prediction"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:Purple; border:0; color:white' role="tab" aria-controls="home"><center>Prediction</center></h2>

# ### Prepare test data

# In[34]:


target.head()


# In[35]:


df_target = target


# In[36]:


# Remove stopwords & convert to lower case
df_target['text'] = df_target['text'].apply(lambda x: ' '.join([w for w in str(x).lower().split() if w not in STOPWORDS]))

# Remove Punctuations
df_target["text"] = df_target['text'].str.replace('[^\w\s]','')
df_target.head()


# In[37]:


target_seq = tokenizer.texts_to_sequences(df_target.text.values)
target_padded = pad_sequences(
    target_seq, maxlen=MAX_LEN, dtype='int32', padding='post',
    truncating='post'
)


# ### Predict

# In[38]:


result = model.predict(target_padded)


# In[39]:


sample


# In[40]:


target['score'] = result


# In[41]:


target[['comment_id','score']].to_csv('./submission.csv', index=False)


# <a id="references"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:Purple; border:0; color:white' role="tab" aria-controls="home"><center>References</center></h2>

# 1. EDA - [NLP with Disaster Tweets - EDA, Cleaning and BERT](https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert) 
# 2. Notebook formating - [[V7]Shopee InDepth EDA:One stop for all your needs
# ](https://www.kaggle.com/ishandutta/v7-shopee-indepth-eda-one-stop-for-all-your-needs)  
# 3. [Simple LSTM With Word2Vec](https://www.kaggle.com/khkuggle/simple-lstm-with-word2vec)  
# 
# Thank you :)
