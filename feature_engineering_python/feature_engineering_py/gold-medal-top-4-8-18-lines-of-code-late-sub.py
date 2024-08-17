#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from catboost import CatBoostClassifier

train = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
test = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
sample = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv')
greeks = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/greeks.csv')
greeks['epsilon_date'] = pd.to_datetime(greeks['Epsilon'].replace('Unknown', None), format='%m/%d/%Y')

# mega feature engineering )
train.EJ = train.EJ.eq('B').astype('int')
test.EJ = test.EJ.eq('B').astype('int')

# just use all features
features = [i for i in train.columns if i not in ['Id', 'Class']]

# drop null and old data
train = train.merge(greeks, on='Id')
train = train[train['epsilon_date'].dt.year >= 2017]


# In[2]:


# scale_pos_weight=15 worked better on my CV than auto_class_weight='Balanced'
model = CatBoostClassifier(scale_pos_weight=15, random_seed=1, verbose=0)
model.fit(train[features], train['Class'])


# In[3]:


submission = pd.DataFrame(test["Id"], columns=["Id"])
submission["class_1"] = model.predict_proba(test[features])[:, 1]
submission["class_0"] = 1 - submission["class_1"]
submission.to_csv('submission.csv', index=False)

