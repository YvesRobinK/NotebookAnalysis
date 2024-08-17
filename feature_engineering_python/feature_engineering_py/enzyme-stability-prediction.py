#!/usr/bin/env python
# coding: utf-8

# ![Untitled.png](attachment:36f88fda-eb01-4dda-a74a-58b84d111cd5.png)

# # 1. Introduction
# <p style="font-family:arial; font-weight:bold; letter-spacing: 2px; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid purple">

# Food product packaging tells us how much protein each product contains. It would be easy to think that protein was one just substance. In fact there are numerous different proteins. Each protein is made of some combination of just 20 amino acids. Some proteins have more specialised roles as biological catalysts, or **enzymes**, and these proteins lie at the heart of this challenge.
# 
# In this notebook I am going to look primarily at what can be gleaned from the large **training set** of enzymes. If we can put together a model which helps us understand their characteristics and predict unknown cases, this may be of value quite apart from its application to the test data, which has very special characteristics.
# 
# The main idea behind the competition is that enzymes have specific 3-D structures which enable them to perform their role. At certain temperatures these structures break down or become denatured and this is what we are asked to predict.
# 

# # 2. Import data and packages
# <p style="font-family:arial; font-weight:bold; letter-spacing: 2px; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid purple">

# Import relevant packages.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import spearmanr


# Import data and example submission file.

# In[2]:


df_train = pd.read_csv('../input/novozymes-enzyme-stability-prediction/train.csv', index_col = 'seq_id')
df_test = pd.read_csv('../input/novozymes-enzyme-stability-prediction/test.csv', index_col = 'seq_id')
train_updates = pd.read_csv('../input/novozymes-enzyme-stability-prediction/train_updates_20220929.csv')
submission = pd.read_csv('../input/novozymes-enzyme-stability-prediction/sample_submission.csv', index_col = 'seq_id')


# Drop the data source column for train and test datasets.

# In[3]:


df_train.drop(columns = 'data_source', inplace = True)
df_test.drop(columns = 'data_source', inplace = True)


# # 3. Data preparation
# <p style="font-family:arial; font-weight:bold; letter-spacing: 2px; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid purple">

# Each row represents a sequence of amino acids which make up the protein/enzyme. We have two other features in the train data, the pH and the temperature at which the enzyme breaks down - this is our measure of thermostability.

# In[4]:


df_train.head()


# The file with training updates df_updates provides some updated values for the training set. All but 25 of these seem to be null values. The simplest approach is to remove them all.

# In[5]:


train_updates.info()


# In[6]:


for seq_id in train_updates.seq_id:
    df_train = df_train.drop(index = seq_id)


# There are 286 records where the pH is not known. Let's leave them in as Xgboost can handle missing values.

# In[7]:


print(df_train.isnull().sum())


# Now let's order the dataset in descending thermostability so that the index number provides us with a measure of descending thermostability rank. (We can dispense with the seq_id column).

# In[8]:


df_train.sort_values('tm',ascending = False, inplace = True)
df_train.reset_index(inplace = True)
df_train.drop(columns = 'seq_id', inplace = True)


# These are the ten most thermostable proteins in the dataset.

# In[9]:


df_train.head(10)


# # 4. Feature engineering
# <p style="font-family:arial; font-weight:bold; letter-spacing: 2px; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid purple">

# We need to find some way of converting the strings of amino acids into usable features. Let's start by adding a column to indicate the length of each protein.

# In[10]:


list = []
for number in range(len(df_train)):
    list.append(len(df_train.protein_sequence[number]))
df_train['length'] = list


# In[11]:


list = []
for number in range(len(df_test)):
    list.append(len(df_test.protein_sequence.iloc[number]))
df_test['length'] = list


# Using the list of 20 amino acids, let's create a feature for each one containing their frequency of occurence.

# In[12]:


aminos = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
for letter in aminos:
    df_train[letter] = df_train.protein_sequence.str.count(letter)
    df_test[letter] = df_test.protein_sequence.str.count(letter)


# The dataset now looks like this.

# In[13]:


df_train.head()


# Rather than train on all the data, let's use a proportion with an existing level of reasonable thermostability - say 55 degrees. This still gives us nearly 7,000 proteins for our model.

# In[14]:


df_train = df_train[df_train.tm>55]
df_train.info()


# # 5. Fitting the model
# <p style="font-family:arial; font-weight:bold; letter-spacing: 2px; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid purple">

# In[15]:


X = df_train.drop(columns = ['tm','protein_sequence'])
y = df_train['tm']


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 11)


# In[17]:


# parameters tuned separately
model1 = xgb.XGBRegressor(n_estimators = 170, max_depth = 5)
model1.fit(X_train, y_train)
predictions1 = model1.predict(X_test)


# In[18]:


print('Mean Absolute Error =', mean_absolute_error(y_test, predictions1))
print('Mean Absolute Percentage Error = ', mean_absolute_percentage_error(y_test, predictions1))


# The Spearman Correlation Coefficient is the main metric for this competition.

# In[19]:


rho, p = spearmanr(y_test, predictions1)
print('Spearman Correlation Coefficient =', rho.round(3))


# # 6. Feature importances
# <p style="font-family:arial; font-weight:bold; letter-spacing: 2px; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid purple">

# In[20]:


feature_imp = pd.Series(model1.feature_importances_,index=X.columns).sort_values(ascending=False)
feature_imp


# In[21]:


sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Feature Importance")
plt.legend()
plt.show()


# # 7. The test data
# <p style="font-family:arial; font-weight:bold; letter-spacing: 2px; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid purple">

# The main features in our model are the individual amino acids. The test set consists of just one protein with lots of small mutations, so there is very little variation in the numbers of individual amino acids.

# In[22]:


df_test


# There are only 2 lengths to the amino acid chain.

# In[23]:


df_test.length.value_counts()


# The pH feature is of no help since every value in the test set is 8.

# In[24]:


df_test.pH.value_counts()


# Our model does not perform particularly well on predicting the thermostability of the test data set for these reasons.

# # 8. Conclusion
# <p style="font-family:arial; font-weight:bold; letter-spacing: 2px; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid purple">

# The model is a relatively simple one, which still has some value as a quick and straightforward predictor of enzyme thermostability.
