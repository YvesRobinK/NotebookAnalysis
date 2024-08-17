#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Data & Data Preprocessing

# In[2]:


train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")


# In[3]:


train.head()


# In[4]:


train['len1'] = train['text'].apply(lambda x:len(str(x).split(' ')))
train['len2'] = train['selected_text'].apply(lambda x:len(str(x).split(' ')))

train.head()


# In[5]:


train['len1'].max()


# In[6]:


submission.head()


# In[7]:


test.head()


# # Feature Engineering - Preparing Training & Test Data

# In[8]:


train['sentiment'] = train['sentiment'].apply(lambda x: 2 if x == 'positive' else 1 if x == 'neutral' else 0)
train.head()


# Get starting locations of `selected_text` in `text`:

# In[9]:


selected_texts = train['selected_text'].astype(str)
all_train_texts = train['text'].astype(str)
text_locations = [all_train_texts[i].find(s) for i, s in enumerate(selected_texts)]


# In[10]:


text_locations[:5]


# In[11]:


train['text_location'] = text_locations
train.head()


# In[12]:


test['sentiment'] = test['sentiment'].apply(lambda x: 2 if x == 'positive' else 1 if x == 'neutral' else 0)
test['len1'] = test['text'].apply(lambda x:len(str(x).split(' ')))
test.head()


# In[13]:


# to predict 'len2'
Y_train1 = train['len2']
X_train1 = train[['sentiment', 'len1']]
X_test = test[['sentiment', 'len1']]

# to predict 'text_location'
Y_train2 = train['text_location']
X_train2 = train[['sentiment', 'len1']]


# In[14]:


Y_train1.head()


# In[15]:


X_train1.head()


# In[16]:


X_test.head()


# # EDA

# ## Mean, Median, Mode of Full Text Lengths

# ### Training Data

# In[17]:


print("The mean length of full text in the training data is " + str(round(train['len1'].mean(), 2)))


# In[18]:


print("The median length of full text in the training data is " + str(round(train['len1'].median(), 2)))


# In[19]:


print("The most common length of full text in the training data is " + str(round(train['len1'].mode()[0], 2)))


# ### Test Data

# In[20]:


print("The mean length of full text in the test data is " + str(round(test['len1'].mean(), 2)))


# In[21]:


print("The median length of full text in the test data is " + str(round(test['len1'].median(), 2)))


# In[22]:


print("The most common length of full text in the test data is " + str(round(test['len1'].mode()[0], 2)))


# ## Mean, Median, Mode of Selected Text Lengths

# In[23]:


print("The mean length of selected text in the training data is " + str(round(train['len2'].mean(), 2)))


# In[24]:


print("The mean length of selected text in the training data is " + str(round(train['len2'].median(), 2)))


# In[25]:


print("The mean length of selected text in the training data is " + str(round(train['len2'].mode()[0], 2)))


# # Predictions With LightGBM

# In[26]:


from sklearn import linear_model
import lightgbm as lgb


# ## Predicting Length of Selected Texts

# In[27]:


reg = lgb.LGBMRegressor()
#reg = linear_model.LinearRegression()
reg.fit(X_train1, Y_train1)


# In[28]:


predicted1 = np.round(reg.predict(X_test))
predicted1[predicted1 < 1] = 1
predicted1


# ## Predicting Starting Location of Selected Text

# In[29]:


reg2 = lgb.LGBMRegressor()
#reg2 = linear_model.LinearRegression()
reg2.fit(X_train2, Y_train2)


# In[30]:


predicted2 = np.round(reg2.predict(X_test))
predicted2[predicted2 < 1] = 1
predicted2


# In[31]:


# now predctions are of the form: index of starting character + length of word
predicted = predicted1 + predicted2
predicted


# ## Some Touch Ups

# In[32]:


sub = test[['textID', 'text']]
sub['preds'] = predicted
sub.head()


# In[33]:


sub['text2'] = sub["text"].apply(lambda x: x.split())
sub


# In[34]:


text2 = sub['text2']
text2


# ## Pinpointing the Selected Text Starting and Endpoints (To Be Refined)

# In[35]:


textx = sub['text'].tolist()
text_sub = [s[int(predicted2.tolist()[ind]):int(predicted2.tolist()[ind])+int(predicted1.tolist()[ind])] for ind, s in enumerate(textx)]


# In[36]:


text_sub[:5]


# ## The Original Pinpointing Method (To Be Refined)

# In[37]:


text2 = [l[-int(predicted.tolist()[ind]):] for ind, l in enumerate(text2)]


# In[38]:


text2[:5]


# In[39]:


sub['text22'] = text2
sub.head()


# In[40]:


sub['result'] = sub["text22"].apply(lambda x: " ".join(x))


# In[41]:


sub.head()


# # Submission

# In[42]:


submission["selected_text"] = sub['result']


# In[43]:


submission.head()


# In[44]:


submission.to_csv('submission.csv', index=False)


# ## This notebook will be periodically updated! Stay tuned and happy Kaggling! :)
