#!/usr/bin/env python
# coding: utf-8

# <a id="1"></a><h1 style='background:#12A4F2; border:0; color:black'><center>1. Introduction</center></h1> 
# 
# 
# In this Notebook, I will:
# * Explore the dataset;
# * Perform feature engineering;
# * Build a baseline model;
# * Prepare a submission
# 
# <a id="0"></a>
# ### Content
# * <a href='#1'>1. Introduction</a>  
# * <a href='#2'>2. Analysis preparation</a>  
# * <a href='#3'>3. Data exploration</a>    
# * <a href='#4'>4. Feature engineering</a>    
# * <a href='#5'>5. Model</a>    
# * <a href='#5'>6. Submission</a>    
# 
# 

# <a id="2"></a><h1 style='background:#12A4F2; border:0; color:black'><center>2. Analysis preparation</center></h1> 

# ## 2.1. Load packages

# In[1]:


import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")


# ## 2.2. Load data

# In[2]:


train_df = pd.read_csv("/kaggle/input/tabular-playground-series-apr-2021/train.csv")
test_df = pd.read_csv("/kaggle/input/tabular-playground-series-apr-2021/test.csv")


# <a id="3"></a><h1 style='background:#12A4F2; border:0; color:black'><center>3. Data exploration</center></h1> 

# ## 3.1. Glimpse the data

# In[3]:


train_df.head()


# In[4]:


test_df.head()


# ## 3.2. Data quality

# In[5]:


train_df.info()


# In[6]:


test_df.info()


# In[7]:


train_df.describe()


# In[8]:


test_df.describe()


# ## 3.3. Data visualization

# In[9]:


def plot_count(feature, title, df, size=1):
    '''
    Plot count of classes / feature
    param: feature - the feature to analyze
    param: title - title to add to the graph
    param: df - dataframe from which we plot feature's classes distribution 
    param: size - default 1.
    '''
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:30], palette='Set1')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()  


# In[10]:


train_df.columns


# In[11]:


plot_count('Survived', 'Survived', train_df, 1.5)


# In[12]:


plot_count('Pclass', 'Pclass - Train', train_df, 1.5)
plot_count('Pclass', 'Pclass - Test', test_df, 1.5)


# In[13]:


plot_count('Sex', 'Sex - Train', train_df, 1.5)
plot_count('Sex', 'Sex - Test', test_df, 1.5)


# In[14]:


plot_count('SibSp', 'SibSp - Train', train_df, 3)
plot_count('SibSp', 'SibSp - Test', test_df, 3)


# In[15]:


plot_count('Parch', 'Parch - Train', train_df, 3)
plot_count('Parch', 'Parch - Test', test_df, 3)


# In[16]:


plot_count('Embarked', 'Embarked - Train', train_df, 2)
plot_count('Embarked', 'Embarked - Test', test_df, 2)


# In[17]:


def plot_feature_distribution(data_df, feature, feature2, title, kde_mode=False, hist_mode=True, log=False):
    f, ax = plt.subplots(1,1, figsize=(12,6))
    for item in list(data_df[feature2].unique()):
        d_df = data_df.loc[data_df[feature2]==item]
        try:
            if log:
                sns.distplot(np.log1p(d_df[feature]), kde=kde_mode, hist=hist_mode, label=item)
            else:
                sns.distplot(d_df[feature], kde=kde_mode, hist=hist_mode, label=item)
        except:
            pass
    plt.legend(labels=list(data_df[feature2].unique()), bbox_to_anchor=(1, 1), loc='upper right', ncol=2)
    plt.title(title)
    plt.show()


# In[18]:


plot_feature_distribution(train_df, 'Age', 'Sex', 'Age distribution, grouped by Sex (Train)')


# In[19]:


plot_feature_distribution(test_df, 'Age', 'Sex', 'Age distribution, grouped by Sex (Test)')


# In[20]:


plot_feature_distribution(train_df, 'Fare', 'Pclass', 'Fare distribution (log), grouped by Pclass (Train)',log=True)


# In[21]:


plot_feature_distribution(test_df, 'Fare', 'Pclass', 'Fare distribution (log), grouped by Pclass (Test)',log=True)


# In[22]:


plt.figure(figsize = (12,12))
plt.title('Features correlation plot (Pearson)')
corr = train_df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="rainbow")
plt.show()


# <a id="4"></a><h1 style='background:#12A4F2; border:0; color:black'><center>4. Feature engineering</center></h1> 

# Mean imputation for few of the features (`Age` and `Fare`).

# In[23]:


train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)
train_df['Fare'].fillna(train_df['Fare'].mean(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)


# One hot encoding for `Sex` and `Embarked` features.

# In[24]:


train_df = pd.get_dummies(train_df, columns=["Sex", "Embarked"])
test_df = pd.get_dummies(test_df, columns=["Sex", "Embarked"])


# In[25]:


predictors = ["Pclass", "Age", "SibSp", "Parch", "Sex_male", "Sex_female", "Embarked_C", "Embarked_Q", "Embarked_S"]
target = 'Survived'


# <a id="5"></a><h1 style='background:#12A4F2; border:0; color:black'><center>5. Model</center></h1> 

# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# In[27]:


trn_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=True)


# In[28]:


clf = RandomForestClassifier(n_jobs=-1, 
                             random_state=42,
                             criterion='gini',
                             n_estimators=200,
                             verbose=False)


# In[29]:


print(f"train/validation shape: {trn_df.shape}, {val_df.shape}")


# In[30]:


clf.fit(trn_df[predictors], trn_df[target].values)


# In[31]:


preds = clf.predict(val_df[predictors])


# In[32]:


print(f"ROC AUC score (validation): {roc_auc_score(val_df['Survived'].values, preds)}")


# <a id="6"></a><h1 style='background:#12A4F2; border:0; color:black'><center>6. Submission</center></h1> 

# In[33]:


pred_test = clf.predict(test_df[predictors])


# In[34]:


submission_df = pd.read_csv("/kaggle/input/tabular-playground-series-apr-2021/sample_submission.csv")
submission_df['Survived'] = pred_test


# In[35]:


submission_df.head()


# In[36]:


submission_df.to_csv("submission.csv", index=False)

