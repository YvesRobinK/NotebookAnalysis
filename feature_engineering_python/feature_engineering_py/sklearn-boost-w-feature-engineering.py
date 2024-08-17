#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble


# In[2]:


train = pd.read_csv('/kaggle/input/playground-series-s3e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e9/test.csv')
original = pd.read_csv('/kaggle/input/predict-concrete-strength/ConcreteStrengthData.csv')
original = original.reset_index()
original['id'] = original['index'] + 10000
original = original.drop(columns = ['index'])
original = original.rename(columns = {'CementComponent ':'CementComponent'})


# In[3]:


train['Water_Cement'] = train['WaterComponent']/train['CementComponent']
train['Coarse_Fine'] = train['CoarseAggregateComponent']/train['FineAggregateComponent']
train['Aggregate'] = train['CoarseAggregateComponent'] + train['FineAggregateComponent']
train['Aggregate_Cement'] = train['Aggregate']/train['CementComponent']
train['Slag_Cement'] = train['BlastFurnaceSlag']/train['CementComponent']
train['Ash_Cement'] = train['FlyAshComponent']/train['CementComponent']
train['Plastic_Cement'] = train['SuperplasticizerComponent']/train['CementComponent']
train['Age_Water'] = train['AgeInDays']/train['WaterComponent']

test['Water_Cement'] = test['WaterComponent']/test['CementComponent']
test['Coarse_Fine'] = test['CoarseAggregateComponent']/test['FineAggregateComponent']
test['Aggregate'] = test['CoarseAggregateComponent'] + test['FineAggregateComponent']
test['Aggregate_Cement'] = test['Aggregate']/test['CementComponent']
test['Slag_Cement'] = test['BlastFurnaceSlag']/test['CementComponent']
test['Ash_Cement'] = test['FlyAshComponent']/test['CementComponent']
test['Plastic_Cement'] = test['SuperplasticizerComponent']/test['CementComponent']
test['Age_Water'] = test['AgeInDays']/test['WaterComponent']

original['Water_Cement'] = original['WaterComponent']/original['CementComponent']
original['Coarse_Fine'] = original['CoarseAggregateComponent']/original['FineAggregateComponent']
original['Aggregate'] = original['CoarseAggregateComponent'] + original['FineAggregateComponent']
original['Aggregate_Cement'] = original['Aggregate']/original['CementComponent']
original['Slag_Cement'] = original['BlastFurnaceSlag']/original['CementComponent']
original['Ash_Cement'] = original['FlyAshComponent']/original['CementComponent']
original['Plastic_Cement'] = original['SuperplasticizerComponent']/original['CementComponent']
original['Age_Water'] = original['AgeInDays']/original['WaterComponent']


# In[4]:


num_cols = train.select_dtypes(include=np.number).columns.tolist()
num_cols.remove('id')
num_cols.remove('Strength')


# In[5]:


corr_cols = num_cols + ['Strength']


# In[6]:


train = pd.concat([train,original])


# In[7]:


sklearn_boost = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    min_samples_split=3,
    max_features="sqrt",
    validation_fraction=0.2,
    n_iter_no_change=5,
    tol=0.01,
    random_state=0,
)
sklearn_boost.fit(train[num_cols], train['Strength'])


# In[8]:


predict = sklearn_boost.predict(test[num_cols])
test['Strength'] = predict


# In[9]:


submission = test[['id','Strength']]
submission.to_csv('submission.csv', index = False)

