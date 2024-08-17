#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

#metrics
from sklearn.metrics import roc_auc_score, roc_curve

#selection
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, GroupKFold
from sklearn.preprocessing import LabelEncoder, Normalizer, KBinsDiscretizer

#models
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

#warnings
import warnings
warnings.filterwarnings('ignore')


sns.set_style('darkgrid')
sns.set(rc={"figure.figsize":(7, 5)})

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # 1. Import

# In[2]:


train = pd.read_csv('/kaggle/input/autismdiagnosis/Autism_Prediction/train.csv')
test = pd.read_csv('/kaggle/input/autismdiagnosis/Autism_Prediction/test.csv')

dfs = [train, test]
for df in dfs:
    df.set_index('ID', inplace=True)
    
train.shape, test.shape


# In[3]:


train.head().T


# # 2. Analysis

# In this part we will look at our data in 2 steps:
# - Feature analysis: number of unique and null values, colums type, values distribution
# - Vizualization: draw plots and try to find regularity in data

# ## 2.1 Feature Analysis

# In[4]:


train.nunique()


# - *age_desc* have only 1 unique value. It's mean that here no information for our feuture model.
# - *age* and *result* have all 800 rows of unique values. We wiil need cut it on bins.
# - *contry_of_res* and *ethnicity* should cut on bins too.

# In[5]:


train.info()


# We don't have null objects explicitly.

# In[6]:


train.describe()


# More that 75% data have 0 Class/ASD. We shoult split data to train and test carefully to not lose 1 Class.

# ## 2.2 Visualization

# In[7]:


target = 'Class/ASD'


# In[8]:


sns.countplot(data=train, x=target)


# **In our train dataset dominate 0 Class as said before.** Then we need find some insights that help our model predict correctly 1 Class. 
# 
# First draw plot that show as relationship between *age* and *result*.

# In[9]:


columns = ['age', 'result']

cols = 3
rows = 1

fig, ax = plt.subplots(figsize=(cols*7, rows*6), nrows=rows, ncols=cols)
axs = ax.flatten()
for ind, column in enumerate(columns):
    ax_ = axs[ind]
    sns.scatterplot(data=train, x=target, y=column, ax=ax_)
#     ax_.set_title(f'Count')

ax_ = axs[-1]
sns.scatterplot(data=train, x=columns[1], y=columns[0], hue=target, ax=ax_)


#  On the right plot we see, that:
#  - all rows with result less than zero have 0 Class
#  - more 1 Class rows is between 10 and 15 results and 0 and 40 ages
#  

# In[10]:


fig, ax = plt.subplots(figsize=(10, 20))
sns.countplot(data=train, y='contry_of_res', hue=target, ax=ax)

ax.set_title('Сomparison Classes by country')


# *United States* and *United Kingdom* have most rows with 1 Class. It's may be important feature.
# 
# Another country have or more values in 0 class, or have low count of rows. 

# In[11]:


fig, ax = plt.subplots(figsize=(20, 6))
sns.countplot(data=train, x='ethnicity', hue=target, ax=ax)
ax.set_title('Сomparison Classes by ethnicity')


# - *White-European* have about the same count of Classes.
# - *?* is missing values. But all another ethnicity include 0 Class mainly like *?-ethnicity*.
# 
# We will create and use only 1 feature "is white-Europeian".

# In[12]:


scores = [col for col in train.columns if '_Score' in col]

cols = 2
rows = round(len(scores) / 2)

fig, ax = plt.subplots(figsize=(rows*5, cols*10), nrows=rows, ncols=cols)
axs = ax.flatten()
for ind, score in enumerate(scores):
    ax_ = axs[ind]
    sns.countplot(data=train, hue=target, x=score, ax=ax_)
    ax_.set_title(score, fontsize=14)
    ax_.set_xlabel('')


# I don't see very imoortat information here.In future will needs look how each score correlate with other features.

# In[13]:


columns = [col for col in train.columns if train[col].dtype == 'object']
columns.remove('ethnicity')
columns.remove('contry_of_res')

cols = 2
rows = round(len(columns) / 2)

fig, ax = plt.subplots(figsize=(rows*7, cols*10), nrows=rows, ncols=cols)
axs = ax.flatten()
for ind, column in enumerate(columns):
    ax_ = axs[ind]
    sns.countplot(data=train, hue=target, x=column, ax=ax_)
    ax_.set_title(column, fontsize=14)
    ax_.set_xlabel('')


# - *gender, jaundice, austim* need convert to binary values.
# - *used_app_before, relation* we don't use in prediction.

# And finnaly let's check 1 assumption: if we summarize all Scores, how it will influence on Classes.
# 
# Create new featture and draw plot.

# In[14]:


train['Sum_Score'] = np.sum(train[scores], axis=1)
test['Sum_Score'] = np.sum(test[scores], axis=1)

sns.countplot(data=train, x='Sum_Score', hue=target)


# As we can see sum between 0 and 5 have 0 Class mostly. But sum more and equal 9 includes innformation about 1 Class.

# # 3. Create new feature

# In[15]:


for t_col, t_bin in {'result':6, 'age':5}.items():
    kbd = KBinsDiscretizer(n_bins=t_bin, encode='ordinal', strategy='uniform')
    kbd.fit(train[[t_col]])
    for df in dfs:
        df[f'{t_col}_bins'] = kbd.transform(df[[t_col]])

for t_col in ['austim', 'jaundice', 'gender']:
    le = LabelEncoder()
    le.fit(train[t_col])
    for df in dfs:
        df[t_col + '_encode'] = le.transform(df[t_col])
        
contries = ['United States', 'United Kingdom', 'India', 'United Arab Emirates' , 'New Zealand']
for df in dfs:
    for country in contries:
        df[f'is_{country}'] = (df.contry_of_res == country).astype('int8')
    
    df['is_WE'] = (df.ethnicity == 'White-European').astype('int8')


# In[16]:


col_use = ['A1_Score'
           , 'A2_Score'
           , 'A3_Score'
           , 'A4_Score'
           , 'A5_Score'
           , 'A6_Score'
           , 'A7_Score'
           , 'A8_Score'
           , 'A9_Score'
           , 'A10_Score'
           , 'jaundice_encode'
           , 'austim_encode'
           , 'Sum_Score'
           , 'result_bins'
           , 'age_bins'
           , 'gender_encode'
           , 'is_United States'
           , 'is_United Kingdom'
           , 'is_India', 'is_United Arab Emirates','is_New Zealand'
           , 'is_WE'
          ]


# Split carefully our train data to valid and train:
# - select 0 and 1 Classes and split it in equal proportion
# - concat train and valid data
# - select X and y dataframes for model

# In[17]:


random_st = 42
size = 0.8
train_0, train_1 = train.loc[train[target] == 0], train.loc[train[target] == 1]

df_train_0, df_valid_0,= train_test_split(train_0, train_size=size, random_state=random_st)
df_train_1, df_valid_1 = train_test_split(train_1, train_size=size, random_state=random_st)
df_train, df_valid = pd.concat([df_train_0, df_train_1]).sample(frac=1), pd.concat([df_valid_0, df_valid_1]).sample(frac=1)

X_train, y_train, X_valid, y_valid = df_train[col_use], df_train[target], df_valid[col_use], df_valid[target]

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# In[18]:


X_train.head().T


# # 4. Model selection

# Choose 1 good model for this competition and check how our features affects the quality of the forecast.

# In[19]:


models = [GaussianNB, LogisticRegression, RandomForestClassifier, SVC, KNeighborsClassifier, GradientBoostingClassifier, XGBClassifier, BernoulliNB]
model_results = pd.DataFrame(columns=['model_name', 'ra_train', 'ra_valid'])

for model in models:
    m = model().fit(X_train, y_train)
    pred_train = roc_auc_score(y_train, m.predict(X_train))
    pred_valid = roc_auc_score(y_valid, m.predict(X_valid))
    
    model_results.loc[len(model_results), ] = [model.__name__, pred_train, pred_valid]

model_results.sort_values('ra_valid', ascending=False)


# In[20]:


for i in range(len(col_use)):
    cols = col_use[:i+1]
    model = BernoulliNB(alpha=9)
    model.fit(X_train[cols], y_train)

    pred_train = roc_auc_score(y_train, model.predict(X_train[cols]))
    pred_valid = roc_auc_score(y_valid, model.predict(X_valid[cols]))

    print(round(pred_train,4), round(pred_valid, 4), cols[i])


# *jaundice_encode, austim_encode, age_bins, gender_encode, is_India, is_United Arab Emirates, is_New Zealand* features not usefull. Drop it and check how our model will predict train and valid data.

# In[21]:


col_use = ['A1_Score'
           , 'A2_Score'
           , 'A3_Score'
           , 'A4_Score'
           , 'A5_Score'
           , 'A6_Score'
           , 'A7_Score'
           , 'A8_Score'
           , 'A9_Score'
           , 'A10_Score'
           , 'Sum_Score'
           , 'result_bins'
           , 'is_United States'
           , 'is_WE'
          ]


# In[22]:


from sklearn.naive_bayes import BernoulliNB

nb = BernoulliNB()
nb.fit(X_train[col_use], y_train)

pred_train = roc_auc_score(y_train, nb.predict(X_train[col_use]))
pred_valid = roc_auc_score(y_valid, nb.predict(X_valid[col_use]))

pred_train, pred_valid


# # 5. Predict

# In[23]:


model = BernoulliNB()

model.fit(train[col_use], train[target])
model.predict(test[col_use])

df_submission = pd.DataFrame(index=test.index, columns=[target])
yhat = nb.predict(test[col_use])

df_submission[target] = yhat
df_submission.to_csv('submission.csv', index=True)


# In[24]:


get_ipython().system('cat /kaggle/working/submission.csv')


# In[ ]:




