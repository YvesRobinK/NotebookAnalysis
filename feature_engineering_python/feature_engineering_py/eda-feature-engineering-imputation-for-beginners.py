#!/usr/bin/env python
# coding: utf-8

# ## *NOTE:* The Notebook is Now updated. 
# 
# **Have a look to find something best out of it**

# # Understand the Data first. It's your weapon

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px  # try plotly, it's amazing

import warnings
warnings.filterwarnings("ignore")


# In[3]:


submission = pd.read_csv('/kaggle/input/widsdatathon2021/SampleSubmissionWiDS2021.csv')
info = pd.read_csv('/kaggle/input/widsdatathon2021/DataDictionaryWiDS2021.csv')
data = pd.read_csv('/kaggle/input/widsdatathon2021/TrainingWiDS2021.csv')
testdf = pd.read_csv('/kaggle/input/widsdatathon2021/UnlabeledWiDS2021.csv')


# In[4]:


#set the df to display maximum columns
pd.set_option('display.max_columns',None)
test = testdf.copy()


# In[5]:


train = data.copy()
print("Training data shape", train.shape)
print("Testing data shape", test.shape)


# In[6]:


# what data types how many cols
train.dtypes.value_counts()


# In[7]:


# have a look over data and it's columns
train.head()


# In[8]:


# we do not need any of id values yet so, drop it.
train.drop(['Unnamed: 0','encounter_id','hospital_id','icu_id'], axis=1, inplace=True)

# drop the readmission_status because of constant value as 0
train.drop('readmission_status',axis=1,inplace=True)


# In[9]:


#let's first save our target variable as target, no need to again again write it big.
target = 'diabetes_mellitus'


# In[10]:


train[target].value_counts().plot(kind="pie", explode=[0,0.1], autopct="%.2f", labels=["No","Yes"])
plt.show()


# **The Dataset we are having is IMBALANCED dataset, It nedd to be balnced means there should be equal distribution of each kind of data.**

# <h3 style = "background: lightblue":>Categorical Variables</h3>
# - let's start with that

# In[11]:


categorical_features = [feature for feature in train.columns if train[feature].dtype == 'O']
train[categorical_features].head()


# In[12]:


# Gender
train.gender.value_counts()


# In[13]:


# gender with respect to target variable
fig = px.histogram(train, x='gender', y=target, title="Gender wrt diabetes_mellitus", width=600, height=450)
fig.show()


# In[14]:


#let's see te age distribution of Male and Female
#pd.crosstab(train['age'],train['gender'])

plt.rcParams['figure.figsize'] = (7,5)
train.groupby('gender')['age'].plot(kind="kde")
plt.xlabel("age")
plt.legend(loc="best")
plt.show()


# In[15]:


train['ethnicity'].value_counts()


# In[16]:


#ethnicity
fig = px.histogram(train, x='ethnicity', y=target, title="ethnicity wrt diabetes_mellitus", width=600, height=420)
fig.show()


# In[17]:


train['icu_stay_type'].value_counts()


# In[18]:


fig = px.histogram(train, x='icu_stay_type', y=target, width=600, height=450, title="icu_stay_type wrt target")
fig.show()


# In[19]:


# icu_type
train['icu_type'].value_counts()


# In[20]:


fig = px.histogram(train,x='icu_type', y=target, width=600, height=400)
fig.show()


# In[21]:


#hospital_admit_source
#train['hospital_admit_source'].nunique()
train['icu_admit_source'].unique()


# In[22]:


train['hospital_admit_source'].value_counts()


# In[23]:


# how many missing are there in categorical cols
train[categorical_features].isnull().sum()


# **We have to treat this missing values, before encoding. we will find the best possible technique to impute this missing values.**

# In[24]:


# gender has only 66 NaN so, we can impute using Mode
train['gender'].fillna(train['gender'].mode()[0], inplace=True)


# In[25]:


# we can apply random sample imputation in Ethnicity to maintain the distribution
def impute_random(train, col):
    random_sample = train[col].dropna().sample(train[col].isnull().sum(), random_state=0)
    random_sample.index = train[train[col].isnull()].index
    train.loc[train[col].isnull(), col] = random_sample
    
impute_random(train, 'ethnicity')


# In[26]:


# to preserve the distribution let's try random sample imputation for icu_admit_source
impute_random(train, 'icu_admit_source')

#near about 25% of values are missing so, for now let's do random imputation.
impute_random(train, 'hospital_admit_source')


# ## Numerical Features

# In[27]:


numerical_features = [feature for feature in train.columns if train[feature].dtype != 'O']
train[numerical_features].shape


# In[28]:


train[numerical_features].head()


# **We are having a too many feature so we will separate the Binary variable from Numerical to do the clear data analysis and find the relationship between features**

# In[29]:


binary_features = [feature for feature in numerical_features if train[feature].nunique() == 2]
print("Total binary: ", len(binary_features))


# In[30]:


train[binary_features].head(3)


# In[31]:


#impute_random(train,'gcs_unable_apache')
train['gcs_unable_apache'].fillna(train['gcs_unable_apache'].median(), inplace=True)


# In[32]:


#Diabeter with respect to HIV.
pd.crosstab(train[target], train['aids'])


# In[33]:


# train['elective_surgery'].value_counts()
pd.crosstab(train[target], train['elective_surgery'])


# **OBSERVATIONS**
# - near about 23 percent people who undergo elective surgery are suffering from diabetis. 

# <h3 style="background:lightblue">Continous features</h3>
# Except binary features

# In[34]:


cont_features = [feature for feature in numerical_features if feature not in binary_features]
print("total cont feature except binary: ", len(cont_features))


# In[35]:


train[cont_features].head()


# **Did Diabetic Person Weight more?**

# In[36]:


#sns.boxplot(x=target, y="weight", data=train)
train.groupby(target)['weight'].mean().plot(kind="bar")
plt.title("Diabeties wrt Weight")
plt.show()


# In[37]:


# diabetic patient assumes to consume more glucose
train.groupby(target)['glucose_apache'].mean().plot(kind="bar",title="Avg Glucose consumsion of Diabetic person")
plt.show()


# **Diabetic Person weight much more then Non-Diabetic. This could happen because their body consumes lot of glucose from food as we have seen which results in weight gain**

# In[38]:


# for analyzing BMI let's divide the age in different groups
bins = [0,18,30,40,50,60,70,80,120]
labels = ['0-17','18-30','31-40','41-50','51-60','61-70','71-80','80+']
train["age_range"] = pd.cut(train["age"], bins=bins, labels=labels, include_lowest=True)
train.groupby("age_range")["bmi"].mean()


# In[39]:


plt.figure(figsize=(7,5))
sns.barplot(x='age_range', y="bmi", data=train)
plt.title("BMI wrt AGE_Group")
plt.show()


# **BMI Increases upto the age 50 then, it starts decreasing. And on an average overall it's nearby 30. so we can impute with mean or 30**

# In[40]:


# let's see our Hypothesis be tru or not?
sns.lineplot(x=train["glucose_apache"], y=train["weight"])
plt.show()


# **Weight is increasing in a quadratic manner as consumption of glucose is increasing**

# In[41]:


#impute weight.
train.groupby(target)["weight"].mean()


# **Imputation**

# In[42]:


# impute weight wrt target
train["weight"] = np.where(train[target] == 1, train["weight"].fillna(91), train["weight"].fillna(82))

#impute height with 170 as mean and median both
train["height"] = train["height"].fillna(170)

#impute age wrt target with median
train["age"] = np.where(train[target] == 1, train["age"].fillna(66), train["age"].fillna(63))


# In[43]:


#impute bmi
def fill_bmi(df):
    df['bmi'] = np.where(df['bmi'].isnull(), df['weight']/ (df['height']/100)**2, df['bmi'])
    
fill_bmi(train)


# In[44]:


#impute weight.
train.groupby(target)["height"].mean()


# **Instead of having too many features as min and max, what we will do is combine them based on avg (min+max)/2, which will help in easy analysis and features will also reduce.**

# In[45]:


mydata = train[cont_features].copy()
mydata.head(3)


# In[46]:


max_min_features = [feature for feature in cont_features if 'max' in feature or 'min' in feature and feature not in "albumin_apache"]
print("total features: ",len(max_min_features))


# In[47]:


#separate out the max and min features
max_features = [feature for feature in max_min_features if "max" in feature]
min_features = [feature for feature in max_min_features if "min" in feature]
print("max: ",len(max_features))
print("min: ",len(min_features))


# In[48]:


#By some internal error 2 max features are in min_features, remove them manually.
min_features.remove('d1_albumin_max')
min_features.remove('h1_albumin_max')


# In[49]:


# Example: same thing we are going to apply in our data, and like this, we will shape the name of variables.
s = "d1_albumin_max"
"_".join(s.split("_")[:-1]) + "_avg"

#"_".join(min_features[0].split("_")[:-1]) + "_avg"


# In[50]:


# take the average and add them to the data.
for i in range(0,64):
    col = "_".join(min_features[i].split("_")[:-1]) + "_avg"
    avg = (mydata[min_features[i]] +  mydata[max_features[i]]) / 2
    mydata[col] = avg


# In[51]:


# we got our average features, now remove the max and min features from mydata
mydata.drop(max_features, axis=1, inplace=True)
mydata.drop(min_features, axis=1, inplace=True)

# we have reduce a 64 features, which is better then PCA too.
mydata.shape


# **Wow! That's amazing removing all those 64 features and having average, but still there is lots of noise and most important,there is lots of missing missing values in each col so,we have to impute that first.**

# In[52]:


#let's visualize the missing values with help of amazing library missingno
import missingno as msno


# In[53]:


#plot the missing value graph
msno.matrix(mydata.select_dtypes(include=[np.number]))
plt.show()


# **We can clearly observe the quantity of missing values in each numerical column, it's in huge amount. except starting 5 columns which we have imputed early. we will try to impute at best level with proper analysis**

# In[54]:


avg_features = [feature for feature in mydata.columns if 'avg' in feature]
print(len(avg_features))


# In[55]:


#let's ee the percentage of missing values, if per will be less ten 30 per then we will impute it.
miss_nan = mydata[avg_features].isnull().sum() / len(mydata) * 100
miss_nan.sort_values(ascending=False).to_frame().head(10)


# In[56]:


#drop the feature which are having missing greater then 30%
great_30 = miss_nan[miss_nan[:] > 30].index
mydata.drop(great_30, axis=1, inplace=True)
#Now, we will have 59 cols, more 33 we have reduced.


# In[57]:


less_30 = miss_nan[miss_nan[:] < 30].index
# as we have taken the avg, so just fill it with mean now.
for col in less_30:
    mydata[col] = mydata[col].fillna(mydata[col].mean())
    
print("Imputed Succesfully")
print("Null value in avg col: ",mydata[less_30].isna().sum().sum())


# In[58]:


mydata.head(2)


# **Except average features, we are also having some numeric features which we have to impute**

# In[59]:


other_nan = [feature for feature in mydata.columns if "avg" not in feature]
print("total other features then AVG: ", len(other_nan))


# In[60]:


ot_nan = mydata[other_nan].isnull().sum() / len(mydata) *100
ot_nan.sort_values(ascending=False).head()


# In[61]:


remove_nan = ot_nan[ot_nan[:] > 50].index  #remove greater then 50
fill_nan = ot_nan[ot_nan[:] < 50].index  #impute this


# In[62]:


#drop the feature with greater then 50 percent missing values and 
#we will impute remaining, at little risk.
mydata.drop(remove_nan, axis=1, inplace=True)


# In[63]:


# You can also fill mith (max-min)/2. It is also a good. only some point of diff will there.
for col in fill_nan:
    mydata[col] = mydata[col].fillna(mydata[col].mean())


# In[64]:


mydata.isnull().sum().sum()


# ### Concat the mydata back to training data and proceed further

# In[65]:


#first we have to free all the cont features from cont then concat it
train.drop(cont_features,axis=1, inplace=True)
train.drop("age_range", axis=1, inplace=True)


# In[66]:


# concat mydata to train
train = pd.concat([train, mydata], axis=1)


# In[67]:


train.head(2)
#train.shape


# In[68]:


#let's check is there any col remaining to impute
train.isna().sum().sum()


# ## Categorical Encoding

# In[69]:


train[categorical_features].head(2)


# In[70]:


train['gender'] = train['gender'].map({'M':0,'F':1})


# In[71]:


# let's encode the feature using Integer label encoding
# we will use value_counts to know, how it is encoded, you can also use unique()
cat_cols = ['ethnicity', 'hospital_admit_source','icu_admit_source','icu_stay_type','icu_type']
for col in cat_cols:
    map_dict = {k: i for i, k in enumerate(train[col].value_counts().index, 0)}
    train[col] = train[col].map(map_dict)


# In[72]:


train[categorical_features].head(2)


# # Prepare the Test dataset

# In[73]:


# we do not need any of id values yet so, drop it.
test.drop(['Unnamed: 0','encounter_id','hospital_id','icu_id'], axis=1, inplace=True)

# drop the readmission_status because of constant value as 0
test.drop('readmission_status',axis=1,inplace=True)

# Missing value imputation
#random sampleing func for test set
def impute_random_test(test, col):
    random_sample = test[col].dropna().sample(test[col].isnull().sum(), random_state=0)
    random_sample.index = test[test[col].isnull()].index
    test.loc[test[col].isnull(), col] = random_sample
    

#first impute gender, only 5 missing value
test['gender'].fillna("M", inplace=True)

#random sample imputation for ethnicity, icu_admit_source, hospital_admit_source, gcs_unable_apache
impute_random_test(test, 'ethnicity')
impute_random_test(test, 'icu_admit_source')
impute_random_test(test, 'hospital_admit_source')

test['gcs_unable_apache'].fillna(test['gcs_unable_apache'].median(), inplace=True)

#impute bmi, weight, height
test['weight'] = test['weight'].fillna(test['weight'].mean())
test['height'] = test['height'].fillna(170)   #median
#impute bmi
fill_bmi(test)


# In[74]:


mytest = test[cont_features].copy()

# take the average and add them to the data.
for i in range(0,64):
    col = "_".join(min_features[i].split("_")[:-1]) + "_avg"
    avg = (mytest[min_features[i]] +  mytest[max_features[i]]) / 2
    mytest[col] = avg
    
# we got our average features, now remove the max and min features from mydata
mytest.drop(max_features, axis=1, inplace=True)
mytest.drop(min_features, axis=1, inplace=True)

# we have reduce a 64 features, which is better then PCA too.

mytest.drop(great_30, axis=1, inplace=True)

#inpute remaining features have missing less then 30%
for col in less_30:
    mytest[col] = mytest[col].fillna(mytest[col].mean()) 


mytest.drop(remove_nan, axis=1, inplace=True)
#impute features other then avg, categorical, binary.
for col in fill_nan:
    mytest[col] = mytest[col].fillna(mytest[col].mean())

    
#concat the mytest to test, and before that drop all continuous features
test.drop(cont_features, axis=1, inplace=True)

test = pd.concat([test, mytest], axis=1)


# In[75]:


# Categorical Encoding
test['gender'] = test['gender'].map({'M':0,'F':1})

# let's encode the feature using Integer label encoding
# we will use value_counts to know, how it is encoded, you can also use unique()
cat_cols = ['ethnicity', 'hospital_admit_source','icu_admit_source','icu_stay_type','icu_type']
for col in cat_cols:
    map_dict = {k: i for i, k in enumerate(test[col].value_counts().index, 0)}
    test[col] = test[col].map(map_dict)


# ## Ready for Modelling

# In[76]:


#let's try to build a simple model first
from sklearn.model_selection import train_test_split
# let's find the better accuracy and fit it
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# In[77]:


x = train.drop(target, axis=1)
y = train[target]
testing = test[:]


# In[78]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=11)


# In[79]:


log_clf = LogisticRegression()
knn_clf = KNeighborsClassifier()
dt_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier()


# In[80]:


models = [log_clf, knn_clf, dt_clf, rf_clf]

for clf in models:
    clf.fit(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    
    print(clf.__class__.__name__, " accuracy: ", accuracy_score(y_test,y_pred))


# In[81]:


#let's try to implemet xgBoost then, we will improve the dataset 
from xgboost import XGBClassifier

xgb_clf = XGBClassifier()
xgb_clf.fit(x_train, y_train)
xg_pred = xgb_clf.predict(x_test)
print("accuracy: ", accuracy_score(y_test, xg_pred))


# In[82]:


xgb_prediction = xgb_clf.predict_proba(testing)
xgb_prediction = xgb_prediction[:, -1]

xgb_file= testdf[["encounter_id"]]
xgb_file["diabetes_mellitus"]= xgb_prediction

xgb_file.to_csv('xgboost.csv',index=False)
xgb_file.head()


# **Guys! Here I have not shown any hyperparameter tuning and advance modelling. I hope you can do this.**
# 
# ### Thank You. If you find any corrections then please suggest in comment section, it will very much helpful to improve. If you have any furher techniques that should be tried, please put them forward. And Upvote for motivating to move ahead with data science journey.
