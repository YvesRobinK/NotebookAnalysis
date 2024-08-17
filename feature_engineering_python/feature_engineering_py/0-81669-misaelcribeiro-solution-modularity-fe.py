#!/usr/bin/env python
# coding: utf-8

# ### The notebook is a follow-up work by an impressive solution from misaelcribeiro. We're just applying a novelty method of features engineering into the original pipeline.
# 
# ### Thanks for misaelcribeiro's effort and please upvote their notebook.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import Pool, CatBoostClassifier



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <h3 style="text-align: center;">Loading the data</h3>

# In[2]:


train_df = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv', index_col='PassengerId')


# <h3 style="text-align: center;">Some feature engineering</h3>

# In[3]:


train_df.drop('Name', axis=1, inplace=True)
test_df.drop('Name', axis=1, inplace=True)


# In[4]:


train_df['Transported'].replace(False, 0, inplace=True)
train_df['Transported'].replace(True, 1, inplace=True)


# <h3 style="text-align: center;">Let's separate the cabin columns in three new features</h3>

# In[5]:


train_df[['deck','num', 'side']] = train_df['Cabin'].str.split('/', expand=True)
test_df[['deck','num', 'side']] = test_df['Cabin'].str.split('/', expand=True)

train_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)


# In[6]:


object_cols = [col for col in train_df.columns if train_df[col].dtype == 'object' or train_df[col].dtype == 'category']
numeric_cols = [col for col in train_df.columns if train_df[col].dtype == 'float64']

print(f'Object cols -- {object_cols}')
print(f'Numeric cols -- {numeric_cols}')


# <h3 style="text-align: center;">Sum of spent value by passenger, creating a new feature</h3>

# In[7]:


col_to_sum = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

train_df['SumSpends'] = train_df[col_to_sum].sum(axis=1)
test_df['SumSpends'] = test_df[col_to_sum].sum(axis=1)


# <h3 style="text-align: center;">Checking null and object columns</h3>

# In[8]:


null_cols = train_df.isnull().sum().sort_values(ascending=False)
null_cols = list(null_cols[null_cols>1].index)
null_cols


# In[9]:


train_df[object_cols] = train_df[object_cols].astype('category')
test_df[object_cols] = test_df[object_cols].astype('category')


# In[10]:


print(f'Train DF shape: {train_df.shape}')
print(f'Test DF shape: {test_df.shape}')


# <h3 style="text-align: center;">Encoding the categorical variables</h3>

# In[11]:


from sklearn.preprocessing import OrdinalEncoder

oc = OrdinalEncoder()

df_for_encode = pd.concat([train_df, test_df])

df_for_encode[object_cols] = df_for_encode[object_cols].astype('category')

df_for_encode[object_cols] = oc.fit_transform(df_for_encode[object_cols])

del train_df, test_df

train_df = df_for_encode.iloc[:8693, :]
test_df = df_for_encode.iloc[8693: , :]

del df_for_encode

test_df.drop('Transported', inplace=True, axis=1)


# In[12]:


print(f'Train DF shape: {train_df.shape}')
print(f'Test DF shape: {test_df.shape}')


# In[13]:


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


ct = ColumnTransformer([("imp", SimpleImputer(strategy='mean'), null_cols)])
    
train_df[null_cols] = ct.fit_transform(train_df[null_cols])
test_df[null_cols] = ct.fit_transform(test_df[null_cols])



# <h3 style="text-align: center;">Prearing the dataset for modeling</h3>

# In[14]:


X = train_df.copy()
y = X.pop('Transported')


# <h3 style="text-align: center;">A free tool of modularity in features engineering</h3>

# In[15]:


get_ipython().system('pip install headjackai-sdk')
get_ipython().system('pip install pandas --upgrade')


# #### This account has been created for show on the kaggle fourm, 
# #### and you can create a another free one on headjackai.com
# #### if you try to test it on other tasks or create a new features engineering model yourself.

# In[16]:


from headjackai.headjackai_hub import headjackai_hub

#host setting
hj_hub = headjackai_hub('http://www.headjackai.com:9000')
hj_hub.login(username='jimliu_kaggle', pwd='12345678')


# In[17]:


#show the all public knowledge (features engineering model)
hj_hub.knowledgepool_check(public_pool=True)


# In[18]:


#in this case, we select drug_type as our features engineering function and the features from misaelcribeiro picked. 
source = 'drug_type'


best_features = [
 'Spaceship-Titanic-CryoSleep',
 'Spaceship-Titanic-RoomService',
 'Spaceship-Titanic-Spa',
 'Spaceship-Titanic-VRDeck',
 'Spaceship-Titanic-deck',
 'Spaceship-Titanic-side',
 'Spaceship-Titanic-SumSpends',
 'drug_type-Sex']


#run features engineering on hj 
hj_X = hj_hub.knowledge_transform(data=X, target_domain='Spaceship-Titanic', 
                                  source_domain=source,
                                  label='')

hj_ts_X = hj_hub.knowledge_transform(data=test_df, target_domain='Spaceship-Titanic', 
                                  source_domain=source,
                                  label='') 


# In[19]:


print(f'Train DF shape: {X.shape}')
print(f'Train DF shape after hj features engineering: {hj_X.shape}')


# In[20]:


hj_X.head(10)


# In[21]:


model = CatBoostClassifier(verbose=False, eval_metric='Accuracy',random_seed=2222)

model.fit(hj_X[best_features], y)
prediction = model.predict(hj_ts_X[best_features])


# In[22]:


#Prediction
final = pd.DataFrame()
final.index = test_df.index
final['Transported'] = prediction
final['Transported'].replace(0, False, inplace=True)
final['Transported'].replace(1, True, inplace=True)
final.to_csv('submission.csv')


# <h3 style="text-align: center;">Final score so far: 0.81669 -- in progress</h3>
