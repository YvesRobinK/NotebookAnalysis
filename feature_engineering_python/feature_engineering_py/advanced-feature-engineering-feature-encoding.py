#!/usr/bin/env python
# coding: utf-8

# ![](https://cdn-images-1.medium.com/max/1145/1*-ZAg5ftPd7KN_cc4Btk7Ww.png)

# ### Objective 
# 
# Feature engineering is one of the most important aspects in Kaggle competitions and it is the part where one should spend the most time on. The objective of this kernel is to demonstarte different types of feature encoding methods used in contests. It is very common to see categorical features in a dataset. So what is feature encoding? It is the process of transforming a categorical variable into a continuous variable and using them in the model. Lets start with basic and go to advanced methods.
# 
# * One Hot Encoding & Label Encoding 
# * Frequency Encoding
# * Target Mean Encoding
# 

# In[1]:


## Loading packages
import numpy as np
import pandas as pd


# In[2]:


## Loading dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

## Glimpse throught the data
train.head()


# In[3]:


train.describe()


# #### Before we jump in feature encoding, let's go ahead and remove unwanted variables like Cabin and Ticket.

# In[4]:


## Removing dummy variables
train.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)
test.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)


# #### 1. Converting missing values to NaN.
# #### 2. Imputation with Median and Mode for "Age" and "Embarked"

# In[5]:


## Fill missing values with NaN
train = train.fillna(np.nan)
test = test.fillna(np.nan)


# In[6]:


## Check for Null values
train.isnull().sum()


# In[7]:


## Missing Values Imputation
train["Age"].fillna(train["Age"].median(), inplace = True)
train["Embarked"].fillna("S", inplace = True)


# In[8]:


## Lets create a variable called title from the name variable
for name in train["Name"]:
    train["Title"] = train["Name"].str.extract("([A-Za-z]+)\.",expand=True)

title_replacements = {"Mlle": "Other", "Major": "Other", "Col": "Other", "Sir": "Other", "Don": "Other", "Mme": "Other",
          "Jonkheer": "Other", "Lady": "Other", "Capt": "Other", "Countess": "Other", "Ms": "Other", "Dona": "Other"}

train.replace({"Title": title_replacements}, inplace=True)
train.replace({"Title": title_replacements}, inplace=True)


# ### One Hot Encoding & Label Encoding
# 
# Let's say we have ‘eggs’, ‘butter’ and ‘milk’ in a categorical variable
# 
# * **One Hot Encoding** will produce three columns and the presence of a class will be represented in binary format. Three classes are separated out to three different features. The alogirithm is only worried about their presence/absence without making any assumptions of their relationship.
# * **Label Encoding** gives numerical aliases to the classes. So the resultant label enocded feature will have 0,1 and 2. The problem with this approach is that there is no relation between these three classes yet our alogirithm might consider them to be ordered (that is there is some relation between them) maybe 0<1<2 that is ‘eggs’<‘butter’<‘milk’.
# 
# Depending on the variable we should either use one hot encoding or label encoding.
# 

# #### One Hot Encoding

# In[9]:


## subset categorical variables which you want to encode
x = train[['Embarked','Pclass','Title']]

x = pd.get_dummies(x, columns=['Embarked','Pclass','Title'], drop_first=False)
x.head()


# #### We see that all the sub categories in a categorical variable have been converted into binary flags. This type of feature encoding is one hot encoding.

# #### Label encoding

# In[10]:


## subset categorical variables which you want to encode
x = train[['Embarked','Pclass','Title']]

from sklearn.preprocessing import LabelEncoder
x = x.apply(LabelEncoder().fit_transform)
x.head()


# #### We see that all the subcategories in the categorical variable have been given numbered aliases.  

# ### Frequency Encoding
# 
# * Step 1 : Select a categorical variable you would like to transform.
# * Step 2 : Group by the categorical variable and obtain counts of each category.
# * Step 3 : Join it back with the train dataset.
# 

# In[11]:


## sample train dataset
sample_train = train[['Embarked','Pclass','Title']]

## Frequency Encoding title variable
y = sample_train.groupby(['Title']).size().reset_index()
y.columns = ['Title', 'Freq_Encoded_Title']
y.head()


# In[12]:


sample_train = pd.merge(sample_train,y,on = 'Title',how = 'left')
sample_train.head()


# #### We see that all the subcategories in the categorical variable have been given the total number of occurance for that specific category. 

# ### Mean Encoding 
# **Survived** is our dependent variable (DV), so let's look at how we can extract features from it. The following steps are used in **Mean encoding**,
# 
# * Step 1 : Select a categorical variable you would like to transform.
# * Step 2 : Group by the categorical variable and obtain aggregated sum over "survived" variable.
# (total number of 1's for each category in DV)
# * Step 3 : Group by the categorical variable and obtain aggregated count over "survived" variable.
# * Step 4 : Divide the step 2 / step 3 results and join it back with the train.
# 

# In[13]:


sample_train = train[['Title','Survived']]

## Mean encoding 
x = sample_train.groupby(['Title'])['Survived'].sum().reset_index()
x = x.rename(columns={"Survived" : "Title_Survived_sum"})

y = sample_train.groupby(['Title'])['Survived'].count().reset_index()
y = y.rename(columns={"Survived" : "Title_Survived_count"})

z = pd.merge(x,y,on = 'Title',how = 'inner')
z['Target_Encoded_over_Title'] = z['Title_Survived_sum']/z['Title_Survived_count']
z.head()


# #### We see that all the subcategories in the categorical variable are represented as the survival probabilty occuring in that specific category.

# In[14]:


## Joining this back with the sample_train dataset

z = z[['Title','Target_Encoded_over_Title']]

sample_train = pd.merge(sample_train,z,on = 'Title',how = 'left')
sample_train.head()


# What will you do if you want to mean encode a categorical variable using a **continuous variable** instead of a **dichotomous/binary variable**? How will you use mean encoding? There are two methods which can be used for mean encoding continuous variables :  
# 
# 1. Direct method
# 2. k-fold method  

# #### Direct Method
# * Step 1 : Select a categorical variable you would like to transform
# * Step 2 : Select a continuous variable variable.
# * Step 3 : Group by the categorical variable and obtain the aggregated mean over the numeric variable.

# In[15]:


## Direct Method
## TYPE 1
## Selecting title (categorical) and Fare (numeric) from the train dataset

sample_train = train[['Title','Fare']]

## Mean encoding 
x = sample_train.groupby(['Title'])['Fare'].mean().reset_index()
x = x.rename(columns={"Fare" : "Title" +"_Mean_Encoded"})
x.head()


# In[16]:


## Joining this back with the sample_train dataset

sample_train = pd.merge(sample_train,x,on = 'Title',how = 'left')
sample_train.head()


# #### We see that each title is encoded into the mean of Ticket Fare. This is a popularly used feature encoding technique in kaggle competitions. 
# 
# #### But why are these encodings better ?
# 
# * Mean encoding can embody the target in the label whereas label encoding has no correlation with the target.
# * In case of large number of features, mean encoding could prove to be a much simpler alternative.
# * A histogram of predictions using label & mean encoding show that mean encoding tend to group the classes together whereas the grouping is random in case of label encoding
# 
# ![](https://cdn-images-1.medium.com/max/800/1*qwooYKx8rU6h1VDnUCgsNg.png)
# 
# 
# * Even though it looks like mean encoding is Superman, it’s kryptonite is overfitting. The fact that we use target classes to encode for our training labels may leak data about the predictions causing the encoding to become biased. Well we can avoid this by Regularizing. 
# 
# #### Now let's look at how we can reduce this bias.

# #### What is k-fold cross validation?
# 
# Cross-validation is primarily used in machine learning to estimate the skill of a machine learning model on unseen data. Let's say the value of k is 5. You break the train dataset into 5 parts hold out one part as test and run a model using the other 4 parts as train. This is iteratively done such that the model trains through all combinations of the dataset. (refer image below)
# 
# #### How can we use this for mean encoding ?
# Since K-fold strategy holds out some data, it reduces the bias we discussed about earlier and applying the direct method over k-folds will be the best way to do feature encoding. 
# 

# ![](https://cdn-images-1.medium.com/max/1600/1*me-aJdjnt3ivwAurYkB7PA.png)

# In[17]:


## K-Fold Method  
## TYPE 2
## Selecting title (categorical) and Fare (numeric) from the train dataset

x = train[['Embarked','Pclass','Title','Fare']]
cols = ['Embarked','Pclass','Title']

## Loading k-fold from sklearn
import sklearn
from sklearn.model_selection import StratifiedKFold

## 10 fold cv
kf = sklearn.model_selection.KFold(n_splits = 10, shuffle = False) 


# In[18]:


for i in cols: ## Looping through all features   
    x['Mean_Encoded_on'] = np.nan

    for tr_ind, val_ind in kf.split(x):
        X_tr, X_val = x.iloc[tr_ind], x.iloc[val_ind] ## train-test hold out
        x.loc[x.index[val_ind], 'Mean_Encoded_on'] = X_val[i].map(X_tr.groupby(i).Fare.mean())

    x = x.rename(index=str, columns={"Mean_Encoded_on": i +"_K_Encoded"})


# In[19]:


x.head()


# #### This is the best way to reduce bias while using mean encoding. Mean, Median, Min, Max and Sum are also common aggregations people use on kaggle while encoding categorical features. 
# 
# Reference : 
# * Medium Articles 
# 1. https://towardsdatascience.com/encoding-categorical-features-21a2651a065c 
# 2. https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63
# 
# * Analytics Vidhya Discussion 
# 1. https://discuss.analyticsvidhya.com/t/label-encoding-vs-one-hot-encoding-in-machine-learning-model/7411

# ### Thanks for you time. Happy kaggling! 
