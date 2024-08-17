#!/usr/bin/env python
# coding: utf-8

# <strong><center style='font-size:40px;font-family:Georgia;'>Predict survival on the Titanic</center><strong>

# ![](https://images.unsplash.com/photo-1614645169630-a3932d31ba92?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Nnx8VGl0YW5pY3xlbnwwfDB8MHx8&auto=format&fit=crop&w=800&q=60)

# # 📖Read Data

# In[97]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd


# In[98]:


train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
print('Shape of Training Data:',train.shape,'Shape of Testing Data',test.shape)


# In[99]:


train.head()


# In[100]:


test.head()


# # 🧹Data Cleaning

# # 1️⃣Drop useless columns

# In[101]:


train1=train.copy()
test1=test.copy()

train1.drop(columns=['PassengerId','Ticket','Cabin'],inplace=True)
test1.drop(columns=['PassengerId','Ticket','Cabin'],inplace=True)


# In[102]:


def gettitle(name):
    str1=name.split(',')[1]
    str2=str1.split('.')[0]
    str3=str2.strip()
    return str3

for data in [train1,test1]:
    for i in range(data.shape[0]):
        data.Name[i]=gettitle(data.Name[i])

train1.head()


# In[103]:


test1.head()


# # 2️⃣Missing Values

# In[104]:


train1.isna().sum()


# In[105]:


test1.isna().sum()


# In[106]:


train1.Embarked.value_counts(dropna=False)


# In[107]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(train1.Age)
plt.show()


# In[108]:


sns.distplot(train1.Fare)
plt.show()


# In[109]:


train2=train1.copy()
test2=test1.copy()

train2.Embarked.fillna('S',inplace=True)
train2.Age.fillna(train2.Age.median(),inplace=True)

test2.Age.fillna(train2.Age.median(),inplace=True)
test2.Fare.fillna(train2.Fare.mean(),inplace=True)

train2.isna().sum()


# In[110]:


test2.isna().sum()


# # 3️⃣Duplicated Values

# In[111]:


train2.duplicated().sum()


# In[112]:


train2.drop_duplicates(inplace=True)
train2.duplicated().sum()


# # 📊Exploratory Data Analytics

# In[113]:


train2.head()


# # 1️⃣Distribution of Categorical Features

# In[114]:


plt.figure(figsize=(10,5))
n=0
for col in ['Survived','Pclass','Sex','SibSp','Parch','Embarked']:
    n+=1
    plt.subplot(3,2,n)
    sns.countplot(train2[col])
    plt.title(f'Distribution of {col}')
plt.tight_layout()


# In[115]:


plt.figure(figsize=(10,5))
sns.countplot(y=train2.Name)
plt.title('Distribution of Name')
plt.tight_layout()


# # 2️⃣Distribution of Numeric Features

# In[116]:


plt.figure(figsize=(10,5))
n=0
for col in ['Age','Fare']:
    n+=1
    plt.subplot(2,2,n)
    sns.distplot(train2[col])
    plt.title(f'Distribution of {col}')
    n+=1
    plt.subplot(2,2,n)
    sns.boxplot(train2[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()


# # 3️⃣Correlation Plot

# In[117]:


plt.figure(figsize=(10,5))
cor=train2.corr()
sns.heatmap(cor,annot=True,cmap='Blues')
plt.title('Correlation Plot')
plt.tight_layout()


# # 🕹️Feature Engineering

# # 1️⃣One-Hot Encoding

# In[118]:


data=pd.concat([train2,test2],axis=0,ignore_index=True)
data[['Pclass','SibSp','Parch']]=data[['Pclass','SibSp','Parch']].astype(object)
data=pd.get_dummies(data)
data.head()


# # 2️⃣Split the Data

# In[119]:


train=data.iloc[:train2.shape[0],]
test=data.iloc[train2.shape[0]:,1:]

train.shape,test.shape


# In[120]:


train.head()


# In[121]:


test.head()


# # 6️⃣Normalizing Numeric Values

# In[122]:


train.Age=(train.Age-train.Age.mean())/train.Age.std()
train.Fare=(train.Fare-train.Fare.mean())/train.Fare.std()

test.Age=(test.Age-train.Age.mean())/train.Age.std()
test.Fare=(test.Fare-train.Fare.mean())/train.Fare.std()

round(train.Age.var()),round(train.Fare.var())


# # 🚀Model Development

# # 1️⃣Train-Test Spliting

# In[123]:


from sklearn.model_selection import train_test_split

x=train.drop(columns='Survived')
y=train.Survived

xtrain,xval,ytrain,yval=train_test_split(x,y,test_size=0.2,random_state=50,shuffle=True)

xtrain.shape,ytrain.shape,xval.shape,yval.shape


# # 2️⃣Modeling

# In[124]:


from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
lr=model.fit(xtrain,ytrain)

from sklearn.model_selection import cross_val_score
train_score=cross_val_score(lr,xtrain,ytrain,scoring='accuracy',cv=10).mean()
val_score=cross_val_score(lr,xval,yval,scoring='accuracy',cv=10).mean()

print('train score:',train_score,'validation score:',val_score)


# # 🎯Submission

# In[125]:


submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission.head()


# In[126]:


result=lr.predict(test)
result=result.astype(int)
submission.Survived=result
submission.head()


# In[127]:


submission.to_csv('/kaggle/working/submission.csv',index=False)


# # 🫡Thank you for reading!
