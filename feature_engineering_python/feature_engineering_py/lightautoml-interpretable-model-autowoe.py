#!/usr/bin/env python
# coding: utf-8

# #### Please upvote if you find the notebook interesting/useful :)
# 
# # Install [AutoWoe](https://github.com/sberbank-ai-lab/AutoMLWhitebox) library
# 
# This library is a part of [LightAutoML](https://github.com/sberbank-ai-lab/LightAutoML) framework and is used in Whitebox preset, but here we will show how to use it separately

# In[1]:


get_ipython().system('pip install -U autowoe')


# # Imports 

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from autowoe import AutoWoE, ReportDeco


# # Data loading

# In[3]:


INPUT_PATH = '../input/tabular-playground-series-apr-2021/'
train_data = pd.read_csv(INPUT_PATH + 'train.csv')
train_data


# In[4]:


test_data = pd.read_csv(INPUT_PATH + 'test.csv')
test_data


# In[5]:


submission = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
submission


# In[6]:


print('TRAIN TARGET MEAN = {:.3f}'.format(train_data['Survived'].mean()))


# # Extra features creation

# In[7]:


def create_extra_features(data):
    data.Cabin = data.Cabin.map(lambda x: str(x)[0].strip())
    data.Ticket = data.Ticket.map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else np.nan)
    
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    data['FirstName'] = data.Name.map(lambda x: str(x).split(',')[0])
    data['Surname'] = data.Name.map(lambda x: str(x).split(',')[1])
    
    for col in ['Name', 'FirstName', 'Surname']:
        data['Counter_' + col] = data[col].map(data.groupby(col)['PassengerId'].count().to_dict())
        
    data.drop(columns = ['Name', 'Surname'], inplace = True)
    
    return data


all_df = pd.concat([train_data, test_data]).reset_index(drop = True)
all_df = create_extra_features(all_df)
train_data, test_data = all_df[:len(train_data)], all_df[len(train_data):]
print(train_data.shape, test_data.shape)


# In[8]:


train_data.head()


# # Splitting data

# In[9]:


tr_data, val_data = train_test_split(train_data, test_size = 0.2, stratify = train_data['Survived'], random_state = 13)
print(tr_data.shape, val_data.shape)


# # Setup interpretable AutoWoe model
# 
# Here we setup the model with `ReportDeco` decorator - this decorator helps us to build automatic report (see Bonus 2 part)

# In[10]:


auto_woe = AutoWoE(monotonic=False,
                 vif_th=20.,
                 imp_th=0,
                 th_const=32,
                 force_single_split=True,
                 min_bin_size = 0.005,
                 oof_woe=True,
                 n_folds=10,
                 n_jobs=4,
                 regularized_refit=True,
                 verbose=2
        )

auto_woe = ReportDeco(auto_woe)


# # Model training

# In[11]:


get_ipython().run_cell_magic('time', '', 'auto_woe.fit(tr_data, \n             target_name="Survived")\n')


# In[12]:


val_pred = auto_woe.predict_proba(val_data)
print("ACC_SCORE = {:.5f}".format(accuracy_score(val_data['Survived'], (val_pred > 0.5).astype(int))))


# # Bonus 1 - Automatic report generation for trained model

# In[13]:


report_params = {"output_path": "./AUTOWOE_REPORT_Validation",
                 "report_name": "AutoWoE automatic report for Syntanic dataset model",
                 "report_version_id": 1,
                 "city": "Moscow",
                 "model_aim": "Here we want to build a model to solve TPS April 2021 competition",
                 "model_name": "Syntanic_AutoWoE_model",
                 "zakazchik": "Kaggle", # sorry for transliterate russian key here - it means the group that ask you to build this model 
                 "high_level_department": "Google",
                 "ds_name": "Alexander Ryzhkov",
                 "target_descr": "Human survived in Titanic disaster",
                 "non_target_descr": "(Sad news) Human not survived in Titanic disaster"}

auto_woe.generate_report(report_params)


# #### Generated report is [here](./AUTOWOE_REPORT_Validation/autowoe_report.html). P.S. It is interactive - to open subtree click on black triangle on the left of the text.

# # Bonus 2 - Automatic SQL inference query generation for trained model
# 
# As our model is interpretable, we can create SQL query for it automatically. With the help of this query you can receive model predictions inside database without Python at all.
# 
# All you need is setup the `table_name` with the initial data

# In[14]:


print(auto_woe.get_sql_inference_query(table_name = 'TABLE_NAME'))


# # Train on the full train 2 separate models for Sex

# In[15]:


def fit_autowoe(data):
    auto_woe = AutoWoE(monotonic=False,
                     vif_th=20.,
                     imp_th=0,
                     th_const=32,
                     force_single_split=True,
                     min_bin_size = 0.01,
                     oof_woe=True,
                     n_folds=10,
                     n_jobs=4,
                     regularized_refit=True,
                     verbose=2
            )
    auto_woe.fit(data, 
                 target_name="Survived")
    return auto_woe


# In[16]:


male_model = fit_autowoe(train_data[train_data['Sex'] == 'male'])
print('=' * 50)
female_model = fit_autowoe(train_data[train_data['Sex'] == 'female'])


# In[17]:


male_pred = male_model.predict_proba(test_data)
female_pred = female_model.predict_proba(test_data)


# In[18]:


preds = np.where(test_data['Sex'] == 'male', male_pred, female_pred)


# In[19]:


preds


# # Create submissions

# In[20]:


submission['Survived'] = (preds > 0.5).astype(int)
submission.to_csv('AutoWoE_submission.csv', index = False)


# In[21]:


submission['Survived'].mean()


# In[ ]:




