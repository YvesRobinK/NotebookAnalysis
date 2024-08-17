#!/usr/bin/env python
# coding: utf-8

# <a id="section-one"></a>
# ## Introduction

# This sheet introduces attempts to use feature engineering for PS3_E9 to improve the prediction capability

# <a id="section-two"></a>
# ## Importing the required libraries

# In[1]:


import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble


# <a id="section-three"></a>
# ## Importing training, test & original data

# In[2]:


train = pd.read_csv('/kaggle/input/playground-series-s3e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e9/test.csv')
original = pd.read_csv('/kaggle/input/predict-concrete-strength/ConcreteStrengthData.csv')
original = original.reset_index()
original['id'] = original['index'] + 10000
original = original.drop(columns = ['index'])
original = original.rename(columns = {'CementComponent ':'CementComponent'})
train.head()


# In[3]:


print(len(train))
print(len(test))
print(len(original))


# <a id="section-four"></a>
# ## Feature Engineering

# The feature engineering in this sheet has been inspired by the following link:
# 
# https://theconstructor.org/concrete/factors-affecting-strength-of-concrete/6220/#:~:text=Concrete%20strength%20is%20affected%20by,humidity%20and%20curing%20of%20concrete.

# The image below shows how the strength of concrete changes with change in Water/Cement Ratio
# 
# ![image.png](attachment:4442e278-31c7-4160-a687-a9d71ba4938b.png)
# 
# Similarly based on the article above, I have created other ratios. 

# In[4]:


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


# <a id="section-five"></a>
# ## Exploratory Data Analysis

# In[5]:


num_cols = train.select_dtypes(include=np.number).columns.tolist()
num_cols.remove('id')
num_cols.remove('Strength')


# In[6]:


ncols = 3
nrows = int(np.ceil(len(num_cols)/ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
for ax, f in zip(axes.flat, num_cols):
    sns.kdeplot(train[f], color='r', label='train', ax=ax)
    sns.kdeplot(test[f], color='b', label='test', ax=ax)
    sns.kdeplot(original[f], color='g', label='original', ax=ax)
    ax.set_title(f)
    ax.legend()
plt.tight_layout()
plt.show()


# The correlation chart below shows that:
# 
# * Strength of concrete is positively influenced by Cement, Plasticizer, Age, Plastic/Cemment ratio and AgeinDays/Water ratio 
# * Strength of concrete is negatively influenced by Water, Water/Cement ratio,Total Aggregate and Aggregate/Cement ratio

# In[7]:


corr_cols = num_cols + ['Strength']
plt.figure(figsize=(15,15))
sns.heatmap(train[corr_cols].corr(),annot=True)
# plt.savefig("Heatmap.png")
plt.show()


# In[8]:


train = pd.concat([train,original])


# <a id="section-six"></a>
# ## Model Building
# 
# For this first attempt, the model is based on parameters from the following sheet: 
# https://www.kaggle.com/code/nikitagrec/top-1-score-11-75-use-original-data-difference

# In[9]:


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


# <a id="section-seven"></a>
# ## Prediction and Submission
# 

# In[10]:


predict = sklearn_boost.predict(test[num_cols])
test['Strength'] = predict


# In[11]:


submission = test[['id','Strength']]
submission.to_csv('submission.csv', index = False)

