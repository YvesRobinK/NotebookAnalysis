#!/usr/bin/env python
# coding: utf-8

# # Achieving 80% accuracy in easy way.
# 
# **This notebook is a beginner's guide to first kaggle competition, I am using basic functions and libraries to perform my data analysis and fitting model into my data.
# 
# To achieve our goal we need to perform these tasks-
# 
# * Importing data and reading it
# * Exploratory Data Analysis
# * Merging data
# * Feature Engineering(Filling missing data,Creating Columns,Dropping columns,mapping data)
# * Correlation and feature importance
# * Fitting the model
# * Predicting Resullts
# 
# **Please if you find this notebook useful,upvote it and feel free to  copy and edit for use**
# 

# In[65]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[66]:


#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


get_ipython().run_line_magic('matplotlib', 'inline')


# In[67]:


#importing data
#importing and reading data
train=pd.read_csv('../input/titanic/train.csv')
test =pd.read_csv("../input/titanic/test.csv")


# In[68]:


#reading data
train.head()


# In[69]:


test.head()


# **As we can see the Survived column is the dependent feature and is absent in test data set**

# In[70]:


#looking at shape and other information of data
train.shape


# In[71]:


test.shape


# 
# Train has 891 rows and test has 418 rows with one missing column which is Survived

# In[72]:


train.info()


# We can see that columns Age, Embarked, fare and Cabin have null values We can also plot it.

# In[73]:


train.isnull().sum()


# In[74]:


test.isnull().sum()


# 
# 
# **We will use heatmap for plotting the missimg values, What is a heatmap? A heat map (or heatmap) is a graphical representation of data where values are depicted by color. Heat maps make it easy to visualize complex data and understand it at a glance. We'll see how!**

# In[75]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# In[76]:


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='rainbow')


# 
# Here we can see that since heatmap depicts value in colors, more than 80% of the cabin data is misssing and few age rows are also missing.

# # Exploratory Data Anlysis
# *So what basically is exploratory data analysis In data mining, Exploratory Data Analysis (EDA) is an approach to analyzing datasets to summarize their main characteristics, often with visual methods. EDA is used for seeing what the data can tell us before the modeling task.It may be tedious, boring, and/or overwhelming to derive insights by looking at plain numbers. Exploratory data analysis techniques have been devised as an aid in this situation.*
# 
# 

# 
# **First we'll separate categorical and numerical features in our data set**

# In[77]:


categorical_features=[features for features in train.columns if train[features].dtypes=='O']
categorical_features


# In[78]:


numerical_features=[features for features in train.columns if train[features].dtypes!='O']
numerical_features


# In[79]:


#plot depicting the relation between Pclass(Ticket class) and Passengers Survived
sns.barplot(x="Pclass",y="Survived",data=train)


# In[80]:


#plotting age feature
sns.FacetGrid(train, hue="Survived", size=5) \
   .map(sns.distplot, "Age") \
   .add_legend()
plt.show()


# In[81]:


#plot depicting the relation between Gender and Passengers Survived
sns.barplot(x="Sex",y="Survived",data=train)


# In[82]:


#plot depicting the relation between Embarked(Port of Embarkation) and Passengers Survived
sns.barplot(x="Embarked",y="Survived",data=train)


# In[83]:


sns.pairplot(train[["Survived","Pclass","Fare","Age"]], hue="Survived", height=3);
plt.show()


# From the above graphs we can easily depict the relation between Survived and mentioned features
# 
# * As most of the first class passengers survived
# * Females Survived more than males
# * People who embarked from "C" had more chances of survival

# 
# **Now we'll merge our data**
# 
# but before mergin we'll seperate out our dependent feature 'Survived'

# In[84]:


y_train= train['Survived']
y_train


# In[85]:


ntrain = train.shape[0]
ntest = test.shape[0]


# In[86]:


all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['Survived'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# In[87]:


all_data.isnull().sum()


# In[ ]:


# **Feature Engineering
After merging data we'll perform our feature engineering on data. We'll perform following 3 things

* fill missing values
* Create new column Family size,fare per person,Title and mapping the categorical features
* Dropping Columns that are not important


# 
# **Filling missing** values Age, fare and Embarked, Cabin have missing values we'll take different approach to fill them , but we'll drop the Cabin feature because it's usually advised to drop feature with high amount of missing values(>50%)

# In[92]:


age_by_pclass_sex =all_data.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
        
print('Median age of all passengers: {}'.format(all_data['Age'].median()))


# In[93]:


all_data['Age']= all_data.groupby(['Sex','Pclass'])['Age'].apply(lambda x:x.fillna(x.median()))
all_data['Age'].isnull().sum()


# *I want to explain this further here, the code might seem a bit overwhelmimg but it simply is grouping age according to the ticket class and gender, and then by using the lamnda function and fillna function I am filling the missing value*

# In[95]:


#filling the missing value in Embarked by mode(most freqquent) value
mode=all_data['Embarked'].mode()
all_data['Embarked']= all_data['Embarked'].fillna('mode')


# In[96]:


#filling fare
med_fare= all_data.groupby(['Pclass','Parch','SibSp']).Fare.median()[3][0][0]
med_fare


# In[97]:


all_data['Fare'] = all_data['Fare'].fillna(med_fare)


# In[98]:


#now checking all the missing value
all_data.isnull().sum()


# now stepping onto the second task that is creating a new column and creating catergorical fetures

# In[99]:


#family size is sum of SibSp(siblings / spouses aboard the Titanic) and Parch(parents / children aboard the Titanic)
all_data['Family_size']= all_data['SibSp']+all_data['Parch']+1


# In[100]:


#creating column title
all_data['Title'] = all_data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
all_data['Title']


# In[101]:


all_data['FarePerPerson']= all_data['Fare']/all_data['Family_size']
all_data['FarePerPerson']


# In[102]:


#dropping columns that are not important
all_data.drop(['Ticket','SibSp','Name','Parch','Cabin'],axis=1,inplace=True)


# In[103]:


all_data.head()


# In[104]:


all_data=all_data.drop(['Fare'],axis=1)


# **Mapping categorical features**
# 
# I will use the label encoder function provided by scikit learn for pre-processing but we can also use mapping function

# In[105]:


categorical_features=[features for features in all_data.columns if all_data[features].dtypes=='O']
categorical_features


# In[106]:


from sklearn.preprocessing import LabelEncoder
# process columns, apply LabelEncoder to categorical features
lbl= LabelEncoder()
lbl.fit(list(all_data['Title'].values)) 
all_data['Title'] = lbl.transform(list(all_data['Title'].values))


# In[107]:


lbl.fit(list(all_data['Sex'].values)) 
all_data['Sex'] = lbl.transform(list(all_data['Sex'].values))


# In[108]:


lbl.fit(list(all_data['Embarked'].values)) 
all_data['Embarked'] = lbl.transform(list(all_data['Embarked'].values))


# In[109]:


all_data.head()


# In[110]:


#seperating data
train = all_data[:ntrain]
test = all_data[ntrain:]


# # Correaltion and feature Importance

# In[111]:


train.corr()


# In[112]:


plt.subplots(figsize=(15,8))

sns.heatmap(train.corr(),annot=True,cmap='Oranges')


# # finally fitting our training set and making prediction

# In[113]:


from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier


# In[114]:


x= train
x


# In[115]:


GBR = GradientBoostingClassifier(n_estimators=100, max_depth=4)
GBR.fit(x,y_train)


# In[116]:


#finalMdG is the prediction by GradientBoostingClassifier
finalMdG=GBR.predict(test)
finalMdG


# In[117]:


ID = test['PassengerId']


# In[118]:


submission=pd.DataFrame()
submission['PassengerId'] = ID
submission['Survived'] = finalMdG
submission.to_csv('submissiongb.csv',index=False)


# In[119]:


rd=RandomForestClassifier()


# In[120]:


rd.fit(x,y_train)


# In[121]:


#finalMdR is the prediction by RandomForestClassifier
finalMdR=rd.predict(test)
finalMdR


# In[122]:


submission=pd.DataFrame()
submission['PassengerId'] = ID
submission['Survived'] = finalMdR
submission.to_csv('submissionrd.csv',index=False)


# # The End 
# I tried my best to keep the code easy and simple I've missed out some of the things like feature scaling and Hyperparameter Tunning but since this is a small dataset the model will work fine.
# 
# **Feel free to comment,quries, suggestion or feedbacks**
# 
# **Humble Request**- *My previous notebook have accidentaly deleted from kaggle(I don't know the reason), it was my first notebook and I managed to gather 30+ upvotes in 2 weeks,*
# **Please upvote my Notebook if you find it useful,and to support**

# In[ ]:




